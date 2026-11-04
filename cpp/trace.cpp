#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();

// Nodes, values, edges and attribute atoms contain only native indices/data.
// Python type objects are interned once; opaque call/external specs are the
// only operation payloads retained as Python objects.
enum class Kind { None, Boolean, Integer, Float, String, Sequence, Mapping, Opaque };
struct Atom { Kind kind; std::uint32_t begin, size; std::uint64_t bits; };
struct Value { std::uint32_t type, producer, index; };
struct Node {
  std::uint32_t opcode, operands, operandCount, results, resultCount;
  std::uint32_t attrs, effects, id, origin;
};
static_assert(std::is_trivial_v<Atom> && std::is_standard_layout_v<Atom>);
static_assert(std::is_trivial_v<Value> && std::is_standard_layout_v<Value>);
static_assert(std::is_trivial_v<Node> && std::is_standard_layout_v<Node>);
struct Type { nb::object object, shape, dtype; };
struct TraceArena;
struct TraceValue { nb::object arena; std::uint32_t value; };
struct TraceOperation { nb::object arena; std::uint32_t node; };
struct TraceHandle { nb::object context; std::uint32_t value; };

struct TraceArena {
  std::vector<Node> nodes;
  std::vector<Value> values;
  std::vector<std::uint32_t> edges, atomEdges, inputs;
  std::vector<Atom> atoms;
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint32_t> stringIds;
  std::vector<Type> types;
  nb::dict typeIds;
  std::vector<nb::object> opaque;
  bool terminated = false;

  TraceArena() {
    nodes.reserve(16); values.reserve(24); edges.reserve(32);
    atoms.reserve(64); atomEdges.reserve(32); inputs.reserve(4);
    types.reserve(4); strings.reserve(16); opaque.reserve(2);
  }

  std::uint32_t string(const std::string &value) {
    auto found = stringIds.find(value);
    if (found != stringIds.end()) return found->second;
    auto id = static_cast<std::uint32_t>(strings.size());
    strings.push_back(value);
    stringIds.emplace(value, id);
    return id;
  }

  std::uint32_t type(nb::handle value) {
    PyObject *found = PyDict_GetItemWithError(typeIds.ptr(), value.ptr());
    if (found) return nb::cast<std::uint32_t>(nb::handle(found));
    if (PyErr_Occurred()) throw nb::python_error();
    const auto id = static_cast<std::uint32_t>(types.size());
    auto shape = nb::hasattr(value, "shape") ? nb::borrow<nb::object>(value.attr("shape")) : nb::none();
    auto dtype = nb::module_::import_("fugatensor.dtypes").attr("DType")(value.attr("dtype"));
    types.push_back({nb::borrow<nb::object>(value), std::move(shape), std::move(dtype)});
    typeIds[value] = nb::int_(id);
    return id;
  }

  std::uint32_t encode(nb::handle value, bool mapping = false) {
    Atom atom{};
    if (mapping || PyDict_Check(value.ptr()) || PyObject_TypeCheck(value.ptr(), &PyDictProxy_Type)) {
      atom.kind = Kind::Mapping;
      nb::object items = nb::steal<nb::object>(PyMapping_Items(value.ptr()));
      if (!items.is_valid()) throw nb::python_error();
      std::vector<std::uint32_t> children;
      for (nb::handle item : nb::borrow<nb::iterable>(items)) {
        children.push_back(encode(item[0]));
        children.push_back(encode(item[1]));
      }
      atom.begin = static_cast<std::uint32_t>(atomEdges.size());
      atom.size = static_cast<std::uint32_t>(children.size());
      atomEdges.insert(atomEdges.end(), children.begin(), children.end());
    } else if (value.is_none()) {
      atom.kind = Kind::None;
    } else if (PyBool_Check(value.ptr())) {
      atom.kind = Kind::Boolean;
      atom.bits = value.ptr() == Py_True;
    } else if (PyLong_Check(value.ptr())) {
      atom.kind = Kind::Integer;
      auto integer = PyLong_AsLongLong(value.ptr());
      if (integer == -1 && PyErr_Occurred()) throw nb::python_error();
      std::memcpy(&atom.bits, &integer, sizeof(integer));
    } else if (PyFloat_Check(value.ptr())) {
      atom.kind = Kind::Float;
      auto number = PyFloat_AS_DOUBLE(value.ptr());
      std::memcpy(&atom.bits, &number, sizeof(number));
    } else if (PyUnicode_Check(value.ptr())) {
      atom.kind = Kind::String;
      atom.begin = string(nb::cast<std::string>(value));
    } else if (PyTuple_Check(value.ptr()) || PyList_Check(value.ptr())) {
      atom.kind = Kind::Sequence;
      std::vector<std::uint32_t> children;
      for (nb::handle child : nb::borrow<nb::iterable>(value)) children.push_back(encode(child));
      atom.begin = static_cast<std::uint32_t>(atomEdges.size());
      atom.size = static_cast<std::uint32_t>(children.size());
      atomEdges.insert(atomEdges.end(), children.begin(), children.end());
    } else {
      atom.kind = Kind::Opaque;
      atom.begin = static_cast<std::uint32_t>(opaque.size());
      opaque.push_back(nb::borrow<nb::object>(value));
    }
    const auto result = static_cast<std::uint32_t>(atoms.size());
    atoms.push_back(atom);
    return result;
  }

  nb::object decode(std::uint32_t index) const {
    const auto &atom = atoms[index];
    switch (atom.kind) {
    case Kind::None: return nb::none();
    case Kind::Boolean: return nb::bool_(atom.bits != 0);
    case Kind::Integer: {
      std::int64_t value;
      std::memcpy(&value, &atom.bits, sizeof(value));
      return nb::int_(value);
    }
    case Kind::Float: {
      double value;
      std::memcpy(&value, &atom.bits, sizeof(value));
      return nb::float_(value);
    }
    case Kind::String: return nb::str(strings[atom.begin].c_str());
    case Kind::Sequence: {
      nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(atom.size));
      for (std::uint32_t i = 0; i < atom.size; ++i)
        PyTuple_SET_ITEM(result.ptr(), i, decode(atomEdges[atom.begin + i]).release().ptr());
      return result;
    }
    case Kind::Mapping: {
      nb::dict result;
      for (std::uint32_t i = 0; i < atom.size; i += 2)
        result[decode(atomEdges[atom.begin + i])] = decode(atomEdges[atom.begin + i + 1]);
      return result;
    }
    case Kind::Opaque: return opaque[atom.begin];
    }
    throw nb::value_error("invalid trace attribute");
  }

  std::uint32_t input(nb::handle model) {
    const auto typeId = type(model), id = static_cast<std::uint32_t>(values.size());
    values.push_back({typeId, absent, static_cast<std::uint32_t>(inputs.size())});
    inputs.push_back(id);
    return id;
  }

  std::uint32_t append(nb::handle self, const std::string &opcode, nb::tuple operands,
                       nb::tuple models, nb::handle attrs, nb::handle effects,
                       nb::handle identifier, nb::handle origin) {
    if (terminated) throw nb::value_error("cannot append operations after a block terminator");
    Node node{string(opcode), static_cast<std::uint32_t>(edges.size()),
              static_cast<std::uint32_t>(nb::len(operands)), static_cast<std::uint32_t>(values.size()),
              static_cast<std::uint32_t>(nb::len(models)), encode(attrs, true), encode(effects),
              encode(identifier), encode(origin)};
    for (nb::handle operand : operands) {
      const auto &value = nb::cast<const TraceValue &>(operand);
      if (!value.arena.is(self)) throw nb::value_error("traced operands must belong to the same graph");
      edges.push_back(value.value);
    }
    std::vector<std::uint32_t> resultTypes;
    for (nb::handle model : models) {
      resultTypes.push_back(type(model));
    }
    const auto index = static_cast<std::uint32_t>(nodes.size());
    for (std::uint32_t i = 0; i < resultTypes.size(); ++i)
      values.push_back({resultTypes[i], index, i});
    nodes.push_back(node);
    terminated = opcode == "return" || opcode == "yield" || opcode == "condition";
    return index;
  }

  nb::tuple emit(nb::handle graph, nb::tuple imported, nb::tuple previous, std::size_t start) const {
    nb::list exported;
    for (std::size_t i = 0; i < values.size(); ++i)
      exported.append(i < nb::len(previous) ? nb::borrow<nb::object>(previous[i]) : nb::none());
    if (nb::len(imported) != inputs.size()) throw nb::value_error("trace import signature changed");
    for (std::size_t i = 0; i < inputs.size(); ++i) exported[inputs[i]] = imported[i];
    for (std::size_t index = start; index < nodes.size(); ++index) {
      const auto &node = nodes[index];
      nb::list operands, resultTypes;
      for (std::uint32_t i = 0; i < node.operandCount; ++i)
        operands.append(exported[edges[node.operands + i]]);
      for (std::uint32_t i = 0; i < node.resultCount; ++i)
        resultTypes.append(types[values[node.results + i].type].object);
      auto operation = graph.attr("operation")(strings[node.opcode], nb::tuple(operands), nb::tuple(resultTypes),
                          "attrs"_a = decode(node.attrs), "effects"_a = decode(node.effects),
                          "id"_a = decode(node.id), "origin_id"_a = decode(node.origin));
      nb::tuple results = nb::cast<nb::tuple>(operation.attr("results"));
      for (std::uint32_t i = 0; i < node.resultCount; ++i) exported[node.results + i] = results[i];
      if (strings[node.opcode] == "return") {
        nb::list returnedTypes;
        for (nb::handle operand : operands) returnedTypes.append(operand.attr("type"));
        graph.attr("result_types") = nb::tuple(returnedTypes);
      }
    }
    return nb::tuple(exported);
  }
};

TraceArena &arena(const TraceValue &value) { return nb::cast<TraceArena &>(value.arena); }
TraceArena &arena(const TraceOperation &value) { return nb::cast<TraceArena &>(value.arena); }
nb::object owner(const TraceHandle &value) { return value.context.attr("_arena"); }

int arenaTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(nb::handle(self))) return 0;
  auto &value = *nb::inst_ptr<TraceArena>(nb::handle(self));
  Py_VISIT(value.typeIds.ptr());
  for (const auto &type : value.types) { Py_VISIT(type.object.ptr()); Py_VISIT(type.shape.ptr()); Py_VISIT(type.dtype.ptr()); }
  for (const auto &object : value.opaque) Py_VISIT(object.ptr());
  return 0;
}
int arenaClear(PyObject *self) {
  auto &value = *nb::inst_ptr<TraceArena>(nb::handle(self));
  value.typeIds.clear(); value.types.clear(); value.opaque.clear();
  return 0;
}
template <typename T> int handleTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (nb::inst_ready(nb::handle(self))) Py_VISIT(nb::inst_ptr<T>(nb::handle(self))->context.ptr());
  return 0;
}
int handleClear(PyObject *self) {
  if (nb::inst_ready(nb::handle(self))) nb::inst_ptr<TraceHandle>(nb::handle(self))->context.reset();
  return 0;
}
template <typename T> int valueTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (nb::inst_ready(nb::handle(self))) Py_VISIT(nb::inst_ptr<T>(nb::handle(self))->arena.ptr());
  return 0;
}
template <typename T> int valueClear(PyObject *self) {
  if (nb::inst_ready(nb::handle(self))) nb::inst_ptr<T>(nb::handle(self))->arena.reset();
  return 0;
}
} // namespace

void bindTrace(nb::module_ &module) {
  static PyType_Slot arenaSlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(arenaTraverse)},
                                   {Py_tp_clear, reinterpret_cast<void *>(arenaClear)}, {0, nullptr}};
  static PyType_Slot handleSlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(handleTraverse<TraceHandle>)},
                                    {Py_tp_clear, reinterpret_cast<void *>(handleClear)}, {0, nullptr}};
  static PyType_Slot valueSlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(valueTraverse<TraceValue>)},
                                   {Py_tp_clear, reinterpret_cast<void *>(valueClear<TraceValue>)}, {0, nullptr}};
  static PyType_Slot operationSlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(valueTraverse<TraceOperation>)},
                                       {Py_tp_clear, reinterpret_cast<void *>(valueClear<TraceOperation>)}, {0, nullptr}};
  nb::class_<TraceArena>(module, "TraceArena", nb::type_slots(arenaSlots))
      .def(nb::init<>())
      .def("input", &TraceArena::input)
      .def("append", [](nb::handle self, const std::string &opcode, nb::tuple operands, nb::tuple types,
                         nb::handle attrs, nb::handle effects, nb::handle id, nb::handle origin) {
        return TraceOperation{nb::borrow<nb::object>(self), nb::cast<TraceArena &>(self).append(
            self, opcode, operands, types, attrs, effects, id, origin)};
      }, "opcode"_a, "operands"_a, "types"_a, "attrs"_a, "effects"_a, "id"_a.none(), "origin"_a.none())
      .def("emit", &TraceArena::emit)
      .def_prop_ro("node_count", [](const TraceArena &self) { return self.nodes.size(); })
      .def_prop_ro("terminated", [](const TraceArena &self) { return self.terminated; });
  nb::class_<TraceValue>(module, "TraceValue", nb::type_slots(valueSlots))
      .def(nb::init<nb::object, std::uint32_t>())
      .def_ro("arena", &TraceValue::arena)
      .def_ro("value_id", &TraceValue::value)
      .def_prop_ro("type", [](const TraceValue &self) { return arena(self).types[arena(self).values[self.value].type].object; })
      .def_prop_ro("producer", [](const TraceValue &self) -> nb::object {
        const auto node = arena(self).values[self.value].producer;
        return node == absent ? nb::none() : nb::cast(TraceOperation{self.arena, node});
      })
      .def_prop_ro("name", [](const TraceValue &self) {
        const auto &value = arena(self).values[self.value];
        return value.producer == absent ? "%arg" + std::to_string(value.index)
             : "%v" + std::to_string(value.producer) + "_" + std::to_string(value.index);
      });
  nb::class_<TraceOperation>(module, "TraceOperation", nb::type_slots(operationSlots))
      .def_prop_ro("opcode", [](const TraceOperation &self) { return arena(self).strings[arena(self).nodes[self.node].opcode]; })
      .def_prop_ro("results", [](const TraceOperation &self) {
        const auto &node = arena(self).nodes[self.node];
        nb::list values;
        for (std::uint32_t i = 0; i < node.resultCount; ++i) values.append(nb::cast(TraceValue{self.arena, node.results + i}));
        return nb::tuple(values);
      });
  nb::class_<TraceHandle>(module, "TraceHandle", nb::type_slots(handleSlots), nb::is_weak_referenceable())
      .def(nb::init<nb::object, std::uint32_t>())
      .def_ro("context", &TraceHandle::context)
      .def_prop_ro("value", [](const TraceHandle &self) { return TraceValue{owner(self), self.value}; })
      .def_prop_ro("shape", [](const TraceHandle &self) {
        auto &a = nb::cast<TraceArena &>(owner(self)); return a.types[a.values[self.value].type].shape;
      })
      .def_prop_ro("dtype", [](const TraceHandle &self) {
        auto &a = nb::cast<TraceArena &>(owner(self)); return a.types[a.values[self.value].type].dtype;
      });
}
