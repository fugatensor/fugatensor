#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {
nb::object checked(PyObject *value) {
  if (!value) throw nb::python_error();
  return nb::steal<nb::object>(value);
}

bool different(nb::handle left, nb::handle right) {
  const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_NE);
  if (result < 0) throw nb::python_error();
  return result != 0;
}

nb::object lookup(nb::dict dictionary, nb::handle key) {
  PyObject *value = PyDict_GetItemWithError(dictionary.ptr(), key.ptr());
  if (!value && PyErr_Occurred()) throw nb::python_error();
  return nb::borrow<nb::object>(value);
}

struct Expression;

struct Tensor {
  nb::object owner, storage, pending, type;
  Tensor *source = nullptr;
  Expression *expression = nullptr;
  std::size_t index = 0;
};

struct Template {
  nb::object owner;
  nb::dict slices, dependencies;
};

struct Slice {
  nb::object owner, key, indices;
  std::vector<std::size_t> inputs;
};

struct Expression {
  nb::object owner;
  nb::tuple inputs, dependencies;
  Template *metadata;
  Slice *selected = nullptr;
  std::vector<std::size_t> required;
};

// All ownership stays in the Python Tensor DAG. These snapshots live only for
// one collection, under the GIL, and each object is decoded at most once.
class Collection {
 public:
  explicit Collection(nb::handle aliasType) : aliasType(aliasType) {}

  nb::tuple collect(nb::sequence roots) {
    std::vector<Tensor *> requested;
    for (nb::handle root : roots) requested.push_back(source(tensor(root)));

    std::vector<Tensor *> pending(requested.rbegin(), requested.rend());
    while (!pending.empty()) {
      Tensor *value = source(pending.back()); pending.pop_back();
      if (!value->storage.is_none()) continue;
      Expression &node = *value->expression;
      const auto index = value->index;
      auto position = std::lower_bound(node.required.begin(), node.required.end(), index);
      if (position != node.required.end() && *position == index) continue;
      node.required.insert(position, index);
      nb::list selected;
      for (auto port : node.required) selected.append(port);
      nb::tuple indices(selected);
      nb::object projection = lookup(node.metadata->slices, indices);
      if (!projection) projection = node.metadata->owner.attr("project")(indices);
      node.selected = &slice(projection);
      for (auto input = node.selected->inputs.rbegin(); input != node.selected->inputs.rend(); ++input)
        pending.push_back(tensor(node.inputs[*input]));
    }

    struct Frame { Tensor *value; Expression *finish; };
    std::vector<Frame> stack;
    for (auto root = requested.rbegin(); root != requested.rend(); ++root)
      stack.push_back({*root, nullptr});
    std::unordered_set<Expression *> complete;
    while (!stack.empty()) {
      const Frame frame = stack.back(); stack.pop_back();
      if (frame.finish) {
        if (complete.insert(frame.finish).second) ordered.push_back(frame.finish);
        continue;
      }
      Tensor *value = source(frame.value);
      if (!value->storage.is_none()) {
        if (leafIndices.emplace(value, leaves.size()).second) leaves.push_back(value);
        continue;
      }
      Expression &node = *value->expression;
      if (!complete.count(&node)) {
        stack.push_back({nullptr, &node});
        for (auto input = node.selected->inputs.rbegin(); input != node.selected->inputs.rend(); ++input)
          stack.push_back({tensor(node.inputs[*input]), nullptr});
      }
    }
    if (leaves.empty()) {
      Tensor *value = carrier(requested);
      leafIndices.emplace(value, 0);
      leaves.push_back(value);
    }

    nb::dict dependencies;
    nb::list expressions;
    nb::dict projections;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
      Expression &node = *ordered[index];
      expressionIndices.emplace(&node, index);
      expressions.append(node.owner);
      projections[node.owner] = node.selected->owner;
      const nb::tuple key = nb::make_tuple(node.selected->indices, node.dependencies);
      nb::object live = lookup(node.metadata->dependencies, key);
      if (!live)
        live = node.metadata->owner.attr("live_dependencies")(node.selected->owner, node.dependencies);
      for (nb::handle pair : nb::borrow<nb::tuple>(live)) {
        const auto entry = nb::borrow<nb::tuple>(pair);
        nb::object previous = lookup(dependencies, entry[0]);
        if (previous && different(previous, entry[1])) changedDefinition();
        dependencies[entry[0]] = entry[1];
      }
    }
    nb::list dependencyItems;
    for (auto [definition, version] : dependencies) {
      if (different(definition.attr("_definition_key")(), version)) changedDefinition();
      dependencyItems.append(nb::make_tuple(definition, version));
    }

    nb::list leafObjects, leafTypes, nodes, outputs;
    for (Tensor *leaf : leaves) {
      leafObjects.append(leaf->owner);
      leafTypes.append(type(leaf));
    }
    for (Expression *node : ordered) {
      nb::list inputs;
      for (auto index : node->selected->inputs) inputs.append(binding(tensor(node->inputs[index])));
      nodes.append(nb::make_tuple(node->selected->key, nb::tuple(inputs)));
    }
    for (Tensor *root : requested) outputs.append(binding(root));
    nb::tuple key = nb::make_tuple(nb::tuple(leafTypes), nb::tuple(nodes), nb::tuple(outputs),
                                  checked(PyFrozenSet_New(dependencyItems.ptr())));
    return nb::make_tuple(leafObjects, expressions, projections, key);
  }

 private:
  nb::handle aliasType;
  std::unordered_map<PyObject *, Tensor> tensors;
  std::unordered_map<PyObject *, Template> templates;
  std::unordered_map<PyObject *, Expression> expressions;
  std::unordered_map<PyObject *, Slice> slices;
  std::vector<Tensor *> leaves;
  std::vector<Expression *> ordered;
  std::unordered_map<Tensor *, std::size_t> leafIndices;
  std::unordered_map<Expression *, std::size_t> expressionIndices;

  [[noreturn]] static void changedDefinition() {
    throw std::runtime_error("a function definition changed after this Tensor was constructed");
  }

  Tensor *tensor(nb::handle owner) {
    auto found = tensors.find(owner.ptr());
    if (found != tensors.end()) return &found->second;
    Tensor value{nb::borrow<nb::object>(owner), owner.attr("_storage"), owner.attr("_pending"), {}};
    return &tensors.emplace(owner.ptr(), std::move(value)).first->second;
  }

  Tensor *source(Tensor *value) {
    std::vector<Tensor *> aliases;
    while (!value->source) {
      if (!value->storage.is_none()) break;
      const int alias = PyObject_IsInstance(value->pending.ptr(), aliasType.ptr());
      if (alias < 0) throw nb::python_error();
      if (!alias) {
        value->expression = &expression(value->pending.attr("expression"));
        value->index = nb::cast<std::size_t>(value->pending.attr("index"));
        break;
      }
      aliases.push_back(value);
      value = tensor(value->pending.attr("source"));
    }
    if (value->source) value = value->source;
    value->source = value;
    for (Tensor *alias : aliases) alias->source = value;
    return value;
  }

  Expression &expression(nb::handle owner) {
    auto found = expressions.find(owner.ptr());
    if (found != expressions.end()) return found->second;
    nb::object metadata = owner.attr("module").attr("_execution_template");
    auto entry = templates.find(metadata.ptr());
    if (entry == templates.end()) {
      Template value{metadata, nb::cast<nb::dict>(metadata.attr("slices")),
                     nb::cast<nb::dict>(metadata.attr("dependencies"))};
      entry = templates.emplace(metadata.ptr(), std::move(value)).first;
    }
    Expression value{nb::borrow<nb::object>(owner), nb::cast<nb::tuple>(owner.attr("inputs")),
                     nb::cast<nb::tuple>(owner.attr("dependencies")), &entry->second, nullptr, {}};
    return expressions.emplace(owner.ptr(), std::move(value)).first->second;
  }

  Slice &slice(nb::handle owner) {
    auto found = slices.find(owner.ptr());
    if (found != slices.end()) return found->second;
    Slice value{nb::borrow<nb::object>(owner), owner.attr("key"), owner.attr("indices"), {}};
    for (nb::handle index : nb::cast<nb::tuple>(owner.attr("inputs")))
      value.inputs.push_back(nb::cast<std::size_t>(index));
    return slices.emplace(owner.ptr(), std::move(value)).first->second;
  }

  Tensor *carrier(const std::vector<Tensor *> &roots) {
    const nb::object preferred = type(roots[0]);
    Tensor *fallback = nullptr;
    std::vector<Tensor *> pending(roots.rbegin(), roots.rend());
    std::unordered_set<Tensor *> seen;
    while (!pending.empty()) {
      Tensor *value = source(pending.back()); pending.pop_back();
      if (!seen.insert(value).second) continue;
      if (!value->storage.is_none()) {
        if (!different(type(value), preferred)) return value;
        if (!fallback) fallback = value;
      } else {
        const auto &inputs = value->expression->inputs;
        for (std::size_t index = inputs.size(); index != 0; --index)
          pending.push_back(tensor(inputs[index - 1]));
      }
    }
    if (!fallback)
      throw nb::value_error("a constant-only deferred graph needs a materialized input carrier");
    return fallback;
  }

  nb::tuple binding(Tensor *value) {
    value = source(value);
    if (!value->storage.is_none()) return nb::make_tuple("input", leafIndices.at(value));
    return nb::make_tuple("result", expressionIndices.at(value->expression), value->index);
  }

  nb::object type(Tensor *value) {
    if (!value->type)
      value->type = nb::make_tuple(value->owner.attr("_shape"), value->owner.attr("_dtype"));
    return value->type;
  }
};
} // namespace

void bindLazy(nb::module_ &module) {
  module.def("collect_pending", [](nb::sequence roots, nb::handle aliasType) {
    if (nb::len(roots) == 0) throw nb::value_error("pending roots must be nonempty");
    return Collection(aliasType).collect(roots);
  }, nb::arg("roots"), nb::arg("alias_type"),
     "Collect the live deferred DAG, validate definitions, and build its exact cache key.");
}
