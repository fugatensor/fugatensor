#include "lazy.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace nb = nanobind;

namespace {
using fuga::lazy::TensorHandle;

void word(std::string &key, std::uint64_t value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}
void text(std::string &key, nb::handle value) {
  Py_ssize_t length;
  const char *bytes = PyUnicode_AsUTF8AndSize(value.ptr(), &length);
  if (!bytes) throw nb::python_error();
  word(key, static_cast<std::uint64_t>(length));
  key.append(bytes, length);
}

struct Record {
  nb::object metadata;
  nb::tuple dependencies;
  bool tupleResult;
};
struct EngineCache {
  nb::object reference;
  std::unordered_map<std::string, Record> records;
};
struct Cache {
  std::unordered_map<PyObject *, EngineCache> engines;
};

class PrimitiveBinder {
 public:
  PrimitiveBinder(nb::object tensorType, nb::object traceType, nb::object dtypeType, nb::object deviceType)
      : tensorType(std::move(tensorType)), traceType(std::move(traceType)), dtypeType(std::move(dtypeType)),
        deviceType(std::move(deviceType)), cache(std::make_shared<Cache>()) {}

  nb::object bind(nb::tuple arguments, nb::handle definition) {
    std::string signature;
    nb::list tensors;
    nb::object engine;
    for (nb::handle argument : arguments) {
      if (nb::isinstance<TensorHandle>(argument)) {
        const auto &tensor = nb::cast<const TensorHandle &>(argument);
        if (!engine.is_valid()) engine = tensor.engine;
        else if (!engine.is(tensor.engine)) engine.attr("_check_tensor")(argument);
        signature.push_back('t');
        word(signature, tensor.typeId());
        tensors.append(argument);
      } else {
        if (nb::isinstance(argument, traceType)) throw nb::type_error("a traced Tensor cannot escape its active trace");
        signature.push_back('s');
        staticKey(signature, argument);
      }
    }
    if (!engine.is_valid()) throw nb::type_error("an operation requires at least one Tensor argument");
    const nb::tuple inputs(tensors);
    // Keep a record alive across definition callbacks, which may clear caches.
    auto foundEngine = cache->engines.find(engine.ptr());
    Record record;
    bool hit = false;
    if (foundEngine != cache->engines.end()) {
      auto found = foundEngine->second.records.find(signature);
      if (found != foundEngine->second.records.end()) {
        record = found->second;
        hit = true;
        for (nb::handle item : record.dependencies) {
          nb::object current = item[0].attr("_definition_key")();
          const auto equal = PyObject_RichCompareBool(current.ptr(), item[1].ptr(), Py_EQ);
          if (equal < 0) throw nb::python_error();
          if (!equal) { hit = false; break; }
        }
      }
    }
    if (!hit) {
      auto traced = definition.attr("_trace_primitive")(arguments, inputs, engine);
      record = {traced.attr("template"), nb::cast<nb::tuple>(traced.attr("dependencies")),
                nb::cast<bool>(traced.attr("is_tuple"))};
      auto position = cache->engines.find(engine.ptr());
      if (position == cache->engines.end()) {
        std::weak_ptr<Cache> weak = cache;
        PyObject *identity = engine.ptr();
        auto callback = nb::cpp_function([weak, identity](nb::handle) {
          if (auto owner = weak.lock()) owner->engines.erase(identity);
        });
        nb::object reference = nb::steal<nb::object>(PyWeakref_NewRef(engine.ptr(), callback.ptr()));
        if (!reference.is_valid()) throw nb::python_error();
        position = cache->engines.emplace(engine.ptr(), EngineCache{std::move(reference), {}}).first;
      }
      position->second.records.insert_or_assign(std::move(signature), record);
      engine.attr("_register_jit")(definition);
    }
    nb::tuple outputs = fuga::lazy::bindPending(engine, tensorType, record.metadata, inputs, record.dependencies);
    return record.tupleResult ? nb::object(outputs) : nb::borrow<nb::object>(outputs[0]);
  }

  void drop(nb::handle engine) { cache->engines.erase(engine.ptr()); }
  void clear() { cache->engines.clear(); }

  int traverse(visitproc visit, void *arg) const {
    Py_VISIT(tensorType.ptr()); Py_VISIT(traceType.ptr()); Py_VISIT(dtypeType.ptr()); Py_VISIT(deviceType.ptr());
    for (const auto &engine : cache->engines) {
      Py_VISIT(engine.second.reference.ptr());
      for (const auto &entry : engine.second.records) {
        Py_VISIT(entry.second.metadata.ptr()); Py_VISIT(entry.second.dependencies.ptr());
      }
    }
    return 0;
  }

 private:
  nb::object tensorType, traceType, dtypeType, deviceType;
  std::shared_ptr<Cache> cache;

  void staticKey(std::string &key, nb::handle value) const {
    if (nb::isinstance(value, dtypeType)) {
      key.push_back('d'); text(key, value.attr("value"));
    } else if (nb::isinstance(value, deviceType)) {
      key.push_back('v'); text(key, value.attr("type")); staticKey(key, value.attr("index"));
    } else if (PyTuple_Check(value.ptr())) {
      key.push_back('q'); word(key, static_cast<std::uint64_t>(nb::len(value)));
      for (nb::handle child : nb::borrow<nb::tuple>(value)) staticKey(key, child);
    } else if (value.is_none()) {
      key.push_back('n');
    } else if (PyBool_Check(value.ptr())) {
      key.push_back(value.ptr() == Py_True ? '1' : '0');
    } else if (PyLong_CheckExact(value.ptr())) {
      int overflow = 0;
      const auto integer = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
      if (integer == -1 && PyErr_Occurred()) throw nb::python_error();
      if (!overflow) {
        key.push_back('i'); word(key, static_cast<std::uint64_t>(integer));
      } else {
        key.push_back('I');
        const auto length = (nb::cast<std::size_t>(value.attr("bit_length")()) + 8) / 8;
        auto bytes = value.attr("to_bytes")(length, "little", nb::arg("signed") = true);
        word(key, length);
        key.append(PyBytes_AS_STRING(bytes.ptr()), length);
      }
    } else if (PyFloat_CheckExact(value.ptr())) {
      key.push_back('f');
      double number = PyFloat_AS_DOUBLE(value.ptr());
      key.append(reinterpret_cast<const char *>(&number), sizeof(number));
    } else if (PyUnicode_CheckExact(value.ptr())) {
      key.push_back('u'); text(key, value);
    } else {
      throw nb::type_error("operation arguments must be Tensor objects or static device/DType/bool/int/float/str/None values or tuples");
    }
  }
};

int traverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  return nb::inst_ready(nb::handle(self)) ? nb::inst_ptr<PrimitiveBinder>(nb::handle(self))->traverse(visit, arg) : 0;
}
int clear(PyObject *self) {
  if (nb::inst_ready(nb::handle(self))) nb::inst_ptr<PrimitiveBinder>(nb::handle(self))->clear();
  return 0;
}
} // namespace

void bindPrimitive(nb::module_ &module) {
  static PyType_Slot slots[] = {{Py_tp_traverse, reinterpret_cast<void *>(traverse)},
                               {Py_tp_clear, reinterpret_cast<void *>(clear)}, {0, nullptr}};
  nb::class_<PrimitiveBinder>(module, "PrimitiveBinder", nb::type_slots(slots))
      .def(nb::init<nb::object, nb::object, nb::object, nb::object>())
      .def("bind", &PrimitiveBinder::bind)
      .def("drop_engine", &PrimitiveBinder::drop)
      .def("clear", &PrimitiveBinder::clear);
}
