#include "lazy.h"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nb = nanobind;
namespace fuga::lazy {
namespace {
using Id = std::uint32_t;
std::size_t mix(std::size_t seed, std::size_t value) {
  return seed ^ (value + std::size_t(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2));
}
nb::object checked(PyObject *value) {
  if (!value) throw nb::python_error();
  return nb::steal<nb::object>(value);
}
bool different(nb::handle left, nb::handle right) {
  const int value = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_NE);
  if (value < 0) throw nb::python_error();
  return value != 0;
}
nb::object lookup(nb::dict dictionary, nb::handle key) {
  PyObject *value = PyDict_GetItemWithError(dictionary.ptr(), key.ptr());
  if (!value && PyErr_Occurred()) throw nb::python_error();
  return nb::borrow<nb::object>(value);
}
std::size_t objectHash(nb::handle value) {
  auto result = PyObject_Hash(value.ptr());
  if (result == -1 && PyErr_Occurred()) throw nb::python_error();
  return std::size_t(result);
}
Py_hash_t pythonHash(std::size_t value) {
  auto result = Py_hash_t(value);
  return result == -1 ? -2 : result;
}
nb::object weakOwner(nb::handle reference) {
#if PY_VERSION_HEX >= 0x030D0000
  PyObject *value = nullptr;
  if (PyWeakref_GetRef(reference.ptr(), &value) < 0) throw nb::python_error();
  return nb::steal<nb::object>(value);
#else
  auto *value = PyWeakref_GetObject(reference.ptr());
  return value == Py_None ? nb::object() : nb::borrow<nb::object>(value);
#endif
}
[[noreturn]] void changedDefinition() {
  throw std::runtime_error("a function definition changed after this Tensor was constructed");
}
struct EngineLock {
  nb::object lock;
  explicit EngineLock(nb::handle engine) : lock(engine.attr("_lock")) { lock.attr("acquire")(); }
  ~EngineLock() {
    PyObject *result = PyObject_CallMethod(lock.ptr(), "release", nullptr);
    if (!result) PyErr_WriteUnraisable(lock.ptr());
    Py_XDECREF(result);
  }
};

struct ValueNode {
  Id references, type, origin, port, alias, storage;
};
struct ExpressionNode {
  Id references, metadata, dependencies, edge, count;
  std::uint64_t epoch;
  std::size_t fingerprint;
};
static_assert(std::is_trivial_v<ValueNode> && std::is_standard_layout_v<ValueNode>);
static_assert(std::is_trivial_v<ExpressionNode> && std::is_standard_layout_v<ExpressionNode>);

struct TypeInfo { nb::tuple shape; nb::object dtype; std::vector<std::int64_t> dimensions; };
struct SliceInfo {
  nb::object owner;
  nb::tuple indices;
  std::vector<Id> ports, inputs;
  Id key = 0;
  std::size_t fingerprint = 0;
};
struct TemplateInfo {
  nb::object owner, reference;
  nb::dict slices = nb::borrow<nb::dict>(nb::handle());
  nb::dict dependencies = nb::borrow<nb::dict>(nb::handle());
  std::vector<Id> outputs;
  Id full = 0, references = 0;
};
struct Binding {
  Id kind, index, port;
  bool operator==(const Binding &other) const {
    return kind == other.kind && index == other.index && port == other.port;
  }
};
struct CanonicalNode {
  Id key;
  std::vector<Binding> inputs;
  bool operator==(const CanonicalNode &other) const { return key == other.key && inputs == other.inputs; }
};
struct ArenaOwner;
struct ExpressionView;
struct ResultView;
struct AliasView;
struct ValueIdentity;
struct PlanKey;
} // namespace

class RuntimeArena : public std::enable_shared_from_this<RuntimeArena> {
 public:
  std::vector<ValueNode> values{1};
  std::vector<ExpressionNode> expressions{1};
  std::vector<Id> edges, freeValues, freeExpressions, freeStorages, freeDependencies, freeTemplates, freeSlices, freeExact;
  std::unordered_map<Id, std::vector<Id>> freeEdges;
  std::vector<nb::object> storages{1};
  std::vector<TypeInfo> types{1};
  std::vector<TemplateInfo> templates{1};
  std::vector<SliceInfo> slices{1};
  std::vector<nb::object> dependencies{nb::tuple()};
  std::vector<nb::object> exactKeys{1};
  std::vector<Id> exactReferences{1};
  nb::dict typeIds, exactIds;
  std::unordered_map<PyObject *, Id> templateIds, sliceIds;
  nb::object tensorType;
  std::uint64_t epoch = 1;
  std::size_t liveValues = 0, liveExpressions = 0, liveEdges = 0;
  bool cleared = false;

  Id exact(nb::handle key) {
    auto found = lookup(exactIds, key);
    if (found) return nb::cast<Id>(found);
    Id id = allocate(exactKeys, freeExact);
    if (exactReferences.size() <= id) exactReferences.resize(id + 1);
    exactKeys[id] = nb::borrow<nb::object>(key);
    exactIds[key] = nb::int_(id);
    return id;
  }
  void retainExact(Id id) { ++exactReferences[id]; }
  void releaseExact(Id id) {
    if (cleared || --exactReferences[id]) return;
    auto key = std::move(exactKeys[id]);
    if (PyDict_DelItem(exactIds.ptr(), key.ptr()) < 0) PyErr_WriteUnraisable(key.ptr());
    freeExact.push_back(id);
  }
  Id type(nb::handle shape, nb::handle dtype) {
    nb::tuple key = nb::make_tuple(shape, dtype);
    auto found = lookup(typeIds, key);
    if (found) return nb::cast<Id>(found);
    TypeInfo info{nb::cast<nb::tuple>(shape),
                  nb::module_::import_("fugatensor.dtypes").attr("DType")(dtype), {}};
    for (nb::handle item : info.shape) info.dimensions.push_back(nb::cast<std::int64_t>(item));
    Id id = Id(types.size());
    types.push_back(std::move(info));
    typeIds[key] = nb::int_(id);
    return id;
  }
  Id dependency(nb::tuple value) {
    if (!nb::len(value)) return 0;
    Id id = allocate(dependencies, freeDependencies);
    dependencies[id] = value;
    return id;
  }
  Id slice(nb::handle owner) {
    if (auto found = sliceIds.find(owner.ptr()); found != sliceIds.end()) return found->second;
    auto key = owner.attr("key");
    Id id = allocate(slices, freeSlices);
    auto weak = weak_from_this();
    auto callback = nb::cpp_function([weak, id, address = owner.ptr()](nb::handle) {
      if (auto arena = weak.lock(); arena && !arena->cleared) {
        arena->sliceIds.erase(address);
        const auto key = arena->slices[id].key;
        arena->slices[id] = {};
        arena->freeSlices.push_back(id);
        arena->releaseExact(key);
      }
    });
    SliceInfo value{checked(PyWeakref_NewRef(owner.ptr(), callback.ptr())),
                    nb::cast<nb::tuple>(owner.attr("indices")), {}, {}, exact(key), objectHash(key)};
    retainExact(value.key);
    for (nb::handle port : value.indices) value.ports.push_back(nb::cast<Id>(port));
    for (nb::handle input : nb::cast<nb::tuple>(owner.attr("inputs"))) value.inputs.push_back(nb::cast<Id>(input));
    slices[id] = std::move(value);
    sliceIds.emplace(owner.ptr(), id);
    return id;
  }
  Id metadata(nb::handle owner) {
    if (auto found = templateIds.find(owner.ptr()); found != templateIds.end()) {
      auto &value = templates[found->second];
      if (!value.owner) {
        value.owner = nb::borrow<nb::object>(owner);
        value.slices = nb::cast<nb::dict>(owner.attr("slices"));
        value.dependencies = nb::cast<nb::dict>(owner.attr("dependencies"));
      }
      return found->second;
    }
    Id id = allocate(templates, freeTemplates);
    auto weak = weak_from_this();
    auto callback = nb::cpp_function([weak, id, address = owner.ptr()](nb::handle) {
      if (auto arena = weak.lock(); arena && !arena->cleared) {
        arena->templateIds.erase(address);
        arena->templates[id] = {};
        arena->freeTemplates.push_back(id);
      }
    });
    TemplateInfo value;
    value.owner = nb::borrow<nb::object>(owner);
    value.reference = checked(PyWeakref_NewRef(owner.ptr(), callback.ptr()));
    value.slices = nb::cast<nb::dict>(owner.attr("slices"));
    value.dependencies = nb::cast<nb::dict>(owner.attr("dependencies"));
    nb::list ports;
    for (nb::handle output : nb::cast<nb::tuple>(owner.attr("output_types"))) {
      value.outputs.push_back(type(output.attr("shape"), output.attr("dtype")));
      ports.append(nb::int_(nb::len(ports)));
    }
    nb::tuple indices(ports);
    nb::object selected = lookup(value.slices, indices);
    if (!selected) selected = owner.attr("project")(indices);
    value.full = slice(selected);
    templates[id] = std::move(value);
    templateIds.emplace(owner.ptr(), id);
    return id;
  }
  template<class Record>
  Id allocate(std::vector<Record> &records, std::vector<Id> &free) {
    if (!free.empty()) { auto id = free.back();free.pop_back();return id; }
    if (records.size() >= std::numeric_limits<Id>::max()) throw std::overflow_error("native graph exhausted uint32 identifiers");
    auto id = Id(records.size());records.emplace_back();return id;
  }
  Id storageSlot(nb::handle storage) {
    auto id = allocate(storages, freeStorages);
    storages[id] = nb::borrow<nb::object>(storage);
    return id;
  }
  Id value(Id typeId, Id origin = 0, Id port = 0, Id alias = 0, nb::handle storage = {}) {
    Id id = allocate(values, freeValues);
    values[id] = {0, typeId, origin, port, alias, storage ? storageSlot(storage) : 0};
    if (origin) ++expressions[origin].references;
    if (alias) retain(alias);
    ++liveValues;
    return id;
  }
  void retain(Id id) { ++values[id].references; }
  void retainExpression(Id id) { ++expressions[id].references; }
  void release(Id id, bool expression = false) {
    if (!id || cleared) return;
    std::vector<std::pair<Id, bool>> pending;
    auto drop = [&](Id index, bool isExpression) {
      auto &references = isExpression ? expressions[index].references : values[index].references;
      if (!--references) pending.emplace_back(index, isExpression);
    };
    drop(id, expression);
    while (!pending.empty()) {
      auto [index, isExpression] = pending.back();pending.pop_back();
      if (isExpression) {
        auto &node = expressions[index];
        const auto metadataId = node.metadata, dependenciesId = node.dependencies;
        for (Id port = 0; port < node.count; ++port) drop(edges[node.edge + port], false);
        freeEdges[node.count].push_back(node.edge);
        liveEdges -= node.count;
        node = {};
        freeExpressions.push_back(index);
        --liveExpressions;
        if (dependenciesId) {
          dependencies[dependenciesId] = {};
          freeDependencies.push_back(dependenciesId);
        }
        auto &metadata = templates[metadataId];
        if (!--metadata.references) {
          auto slices = std::move(metadata.slices);
          auto dependencies = std::move(metadata.dependencies);
          auto owner = std::move(metadata.owner);
        }
      } else {
        auto node = values[index];
        if (node.origin) drop(node.origin, true);
        if (node.alias) drop(node.alias, false);
        values[index] = {};
        freeValues.push_back(index);
        --liveValues;
        if (node.storage) {
          auto storage = std::move(storages[node.storage]);
          freeStorages.push_back(node.storage);
        }
      }
    }
  }
  Id source(Id id) const {
    while (!values[id].storage && values[id].alias) id = values[id].alias;
    return id;
  }
  std::pair<std::size_t, bool> fullHash(Id id) const {
    id = source(id);
    const auto &node = values[id];
    if (node.storage) return {mix(1, node.type), true};
    const auto &expression = expressions[node.origin];
    return {mix(expression.fingerprint, node.port), expression.epoch == epoch};
  }
  Id expression(Id metadataId, const std::vector<Id> &inputs, Id dependenciesId) {
    const auto count = Id(inputs.size());
    Id edge;
    auto &free = freeEdges[count];
    if (!free.empty()) { edge = free.back();free.pop_back(); }
    else {
      if (inputs.size() > std::numeric_limits<Id>::max() - edges.size()) throw std::overflow_error("native graph exhausted uint32 edges");
      edge = Id(edges.size());edges.resize(edges.size() + count);
    }
    for (Id index = 0; index < count; ++index) { edges[edge + index] = inputs[index];retain(inputs[index]); }
    const auto &selected = slices[templates[metadataId].full];
    auto fingerprint = selected.fingerprint;
    bool valid = true;
    for (Id input : selected.inputs) {
      auto child = fullHash(inputs[input]);
      fingerprint = mix(fingerprint, child.first);
      valid &= child.second;
    }
    Id id = allocate(expressions, freeExpressions);
    expressions[id] = {0, metadataId, dependenciesId, edge, count, valid ? epoch : 0, fingerprint};
    ++templates[metadataId].references;
    ++liveExpressions;liveEdges += count;
    return id;
  }
  nb::object wrap(Id id, nb::handle engine, nb::handle type = {}) {
    nb::object owner = nb::inst_alloc(type ? type : nb::handle(tensorType));
    new (nb::inst_ptr<TensorHandle>(owner)) TensorHandle(shared_from_this(), id, nb::borrow<nb::object>(engine));
    nb::inst_mark_ready(owner);
    return owner;
  }
  void publish(Id id, nb::handle storage) {
    auto &node = values[id];
    auto alias = node.alias;
    node.alias = 0;
    if (!node.storage) { node.storage = storageSlot(storage);++epoch; }
    else storages[node.storage] = nb::borrow<nb::object>(storage);
    if (alias) release(alias);
  }
  int traverse(visitproc visit, void *arg) const {
    auto walk = [&](nb::handle object) { return object ? visit(object.ptr(), arg) : 0; };
    for (nb::handle value : {nb::handle(typeIds), nb::handle(exactIds), nb::handle(tensorType)})
      if (int result = walk(value)) return result;
    for (const auto &value : types) { if (int result = walk(value.shape)) return result; if (int result = walk(value.dtype)) return result; }
    for (const auto &value : templates) {
      for (nb::handle object : {nb::handle(value.owner), nb::handle(value.reference), nb::handle(value.slices), nb::handle(value.dependencies)})
        if (int result = walk(object)) return result;
    }
    for (const auto &value : slices) { if (int result = walk(value.owner)) return result; if (int result = walk(value.indices)) return result; }
    for (const auto *objects : {&storages, &dependencies, &exactKeys})
      for (const auto &value : *objects) if (int result = walk(value)) return result;
    return 0;
  }
  void clear() {
    if (cleared) return;
    cleared = true;
    storages.clear();dependencies.clear();templates.clear();slices.clear();types.clear();exactKeys.clear();
    typeIds.clear();exactIds.clear();tensorType = {};
  }
  nb::dict stats() const {
    nb::dict result;
    result["live_nodes"] = liveValues + liveExpressions;
    result["live_values"] = liveValues;result["live_expressions"] = liveExpressions;
    result["live_edges"] = liveEdges;result["value_capacity"] = values.size() - 1;
    result["expression_capacity"] = expressions.size() - 1;
    result["edge_slots"] = edges.size();result["free_value_slots"] = freeValues.size();
    result["value_record_bytes"] = sizeof(ValueNode);
    result["expression_record_bytes"] = sizeof(ExpressionNode);
    result["reserved_node_bytes"] = values.capacity() * sizeof(ValueNode) + expressions.capacity() * sizeof(ExpressionNode);
    result["reserved_edge_bytes"] = edges.capacity() * sizeof(Id);
    result["live_templates"] = templateIds.size();result["template_slots"] = templates.size() - 1;
    result["live_slices"] = sliceIds.size();result["slice_slots"] = slices.size() - 1;
    result["live_dependency_slots"] = dependencies.size() - 1 - freeDependencies.size();
    result["live_exact_keys"] = nb::len(exactIds);
    return result;
  }
};

namespace {
struct ArenaOwner { std::shared_ptr<RuntimeArena> arena = std::make_shared<RuntimeArena>(); };
std::shared_ptr<RuntimeArena> arenaOf(nb::handle engine) {
  return nb::cast<ArenaOwner &>(engine.attr("_arena")).arena;
}
struct Pin {
  std::shared_ptr<RuntimeArena> arena;
  nb::object engine;
  Id id = 0;
  bool expression = false;
  Pin(std::shared_ptr<RuntimeArena> arena, nb::object engine, Id id, bool expression)
      : arena(std::move(arena)), engine(std::move(engine)), id(id), expression(expression) {
    if (expression) this->arena->retainExpression(id);else this->arena->retain(id);
  }
  Pin(const Pin &other) : Pin(other.arena, other.engine, other.id, other.expression) {}
  Pin(Pin &&other) noexcept : arena(std::move(other.arena)), engine(std::move(other.engine)), id(std::exchange(other.id, 0)), expression(other.expression) {}
  ~Pin() { if (arena) arena->release(id, expression); }
  bool same(const Pin &other) const { return arena == other.arena && id == other.id && expression == other.expression; }
  Py_hash_t hash() const { return pythonHash(mix(reinterpret_cast<std::size_t>(arena.get()), id)); }
};
struct ExpressionView : Pin {
  using Pin::Pin;
  nb::object module() const { return arena->templates[arena->expressions[id].metadata].owner.attr("module"); }
  nb::tuple inputs() const {
    const auto node = arena->expressions[id];
    nb::list result;
    for (Id input = 0; input < node.count; ++input) result.append(arena->wrap(arena->edges[node.edge + input], engine));
    return nb::tuple(result);
  }
  nb::tuple dependencies() const { return nb::cast<nb::tuple>(arena->dependencies[arena->expressions[id].dependencies]); }
};
struct ResultView : Pin {
  Id port;
  ResultView(const ExpressionView &owner, Id port) : Pin(owner), port(port) {}
  ResultView(std::shared_ptr<RuntimeArena> arena, nb::object engine, Id expression, Id port)
      : Pin(std::move(arena), std::move(engine), expression, true), port(port) {}
  bool same(const ResultView &other) const { return Pin::same(other) && port == other.port; }
  Py_hash_t hash() const { return pythonHash(mix(std::size_t(Pin::hash()), port)); }
  ExpressionView owner() const { return ExpressionView(arena, engine, id, true); }
};
struct AliasView : Pin {
  using Pin::Pin;
  nb::object source() const { return arena->wrap(id, engine); }
};
struct ValueIdentity : Pin { using Pin::Pin; };

struct PlanKey {
  nb::object owner;
  std::shared_ptr<RuntimeArena> arena;
  std::vector<Id> leaves;
  std::vector<CanonicalNode> nodes;
  std::vector<Binding> outputs;
  Id dependencies;
  std::size_t fingerprint;
  bool pinned = false;
  PlanKey(nb::object owner, std::shared_ptr<RuntimeArena> arena)
      : owner(std::move(owner)), arena(std::move(arena)), dependencies(0), fingerprint(0) {}
  PlanKey(PlanKey &&other) noexcept
      : owner(std::move(other.owner)), arena(std::move(other.arena)), leaves(std::move(other.leaves)),
        nodes(std::move(other.nodes)), outputs(std::move(other.outputs)), dependencies(other.dependencies),
        fingerprint(other.fingerprint), pinned(std::exchange(other.pinned, false)) {}
  ~PlanKey() {
    if (!pinned) return;
    for (const auto &node : nodes) arena->releaseExact(node.key);
    arena->releaseExact(dependencies);
  }
  void pin() {
    for (const auto &node : nodes) arena->retainExact(node.key);
    arena->retainExact(dependencies);
    pinned = true;
  }
  bool same(const PlanKey &other) const {
    return arena == other.arena && dependencies == other.dependencies && leaves == other.leaves &&
           nodes == other.nodes && outputs == other.outputs;
  }
};

class Collection {
 public:
  Collection(std::shared_ptr<RuntimeArena> arena, nb::object engine)
      : arena(std::move(arena)), engine(std::move(engine)) {}
  nb::tuple collect(nb::sequence roots, bool exposeDetails) {
    for (nb::handle root : roots) requested.push_back(arena->source(nb::cast<TensorHandle &>(root).value));
    std::vector<Id> pending(requested.rbegin(), requested.rend());
    while (!pending.empty()) {
      Id value = arena->source(pending.back());pending.pop_back();
      auto node = arena->values[value];
      if (node.storage) continue;
      auto &selection = selections[node.origin];
      auto position = std::lower_bound(selection.required.begin(), selection.required.end(), node.port);
      if (position != selection.required.end() && *position == node.port) continue;
      selection.required.insert(position, node.port);
      const auto expression = arena->expressions[node.origin];
      auto &metadata = arena->templates[expression.metadata];
      if (selection.required.size() == metadata.outputs.size()) {
        selection.slice = metadata.full;
      } else {
        nb::list ports;
        for (Id port : selection.required) ports.append(nb::int_(port));
        nb::tuple indices(ports);
        auto projection = lookup(metadata.slices, indices);
        if (!projection) projection = metadata.owner.attr("project")(indices);
        selection.slice = arena->slice(projection);
      }
      for (auto input = arena->slices[selection.slice].inputs.rbegin(); input != arena->slices[selection.slice].inputs.rend(); ++input)
        pending.push_back(arena->edges[expression.edge + *input]);
    }
    struct Frame { Id value, finish; };
    std::vector<Frame> stack;
    for (auto root = requested.rbegin(); root != requested.rend(); ++root) stack.push_back({*root, 0});
    std::unordered_set<Id> complete;
    while (!stack.empty()) {
      auto frame = stack.back();stack.pop_back();
      if (frame.finish) { if (complete.insert(frame.finish).second) ordered.push_back(frame.finish);continue; }
      Id value = arena->source(frame.value);
      const auto node = arena->values[value];
      if (node.storage) { if (leafIndices.emplace(value, leaves.size()).second) leaves.push_back(value);continue; }
      if (!complete.count(node.origin)) {
        stack.push_back({0, node.origin});
        const auto expression = arena->expressions[node.origin];
        const auto &slice = arena->slices[selections.at(node.origin).slice];
        for (auto input = slice.inputs.rbegin(); input != slice.inputs.rend(); ++input)
          stack.push_back({arena->edges[expression.edge + *input], 0});
      }
    }
    if (leaves.empty()) { Id value = carrier();leafIndices.emplace(value, 0);leaves.push_back(value); }
    nb::dict definitions, projections;
    nb::list expressionObjects, leafObjects;
    PlanKey key(nb::borrow<nb::object>(engine.attr("_arena")), arena);
    for (Id value : leaves) { leafObjects.append(arena->wrap(value, engine));key.leaves.push_back(arena->values[value].type); }
    for (Id id : ordered) {
      auto expression = arena->expressions[id];
      const auto selectedId = selections.at(id).slice;
      auto selectedObject = weakOwner(arena->slices[selectedId].owner);
      auto templateOwner = arena->templates[expression.metadata].owner;
      auto dependencyCache = arena->templates[expression.metadata].dependencies;
      auto dependencies = nb::cast<nb::tuple>(arena->dependencies[expression.dependencies]);
      nb::tuple dependencyKey = nb::make_tuple(arena->slices[selectedId].indices, dependencies);
      auto live = lookup(dependencyCache, dependencyKey);
      if (!live) live = templateOwner.attr("live_dependencies")(selectedObject, dependencies);
      for (nb::handle pair : nb::cast<nb::tuple>(live)) {
        auto entry = nb::cast<nb::tuple>(pair);
        auto previous = lookup(definitions, entry[0]);
        if (previous && different(previous, entry[1])) changedDefinition();
        definitions[entry[0]] = entry[1];
      }
      if (exposeDetails) {
        auto object = nb::cast(ExpressionView(arena, engine, id, true));
        expressionObjects.append(object);projections[object] = selectedObject;
      }
      // Python callbacks may append arena records; only retain IDs across them.
      const auto &selected = arena->slices[selectedId];
      expressionIndices.emplace(id, Id(key.nodes.size()));
      CanonicalNode canonical{selected.key, {}};
      bool full = selectedId == arena->templates[expression.metadata].full;
      for (Id input : selected.inputs) {
        Id value = arena->source(arena->edges[expression.edge + input]);
        canonical.inputs.push_back(binding(value));
        if (!arena->values[value].storage) full &= hashes.at(arena->values[value].origin).second;
      }
      std::size_t fingerprint;
      if (full && expression.epoch == arena->epoch) fingerprint = expression.fingerprint;
      else {
        fingerprint = selected.fingerprint;
        for (Id input : selected.inputs) fingerprint = mix(fingerprint, valueHash(arena->edges[expression.edge + input]));
        if (full) { arena->expressions[id].fingerprint = fingerprint;arena->expressions[id].epoch = arena->epoch; }
      }
      hashes.emplace(id, std::make_pair(fingerprint, full));
      key.nodes.push_back(std::move(canonical));
    }
    nb::list dependencyItems;
    for (auto [definition, version] : definitions) {
      if (different(definition.attr("_definition_key")(), version)) changedDefinition();
      dependencyItems.append(nb::make_tuple(definition, version));
    }
    if (exposeDetails) return nb::make_tuple(leafObjects, expressionObjects, projections);
    auto dependencySet = checked(PyFrozenSet_New(dependencyItems.ptr()));
    key.dependencies = arena->exact(dependencySet);
    key.fingerprint = objectHash(dependencySet);
    for (Id root : requested) { key.outputs.push_back(binding(root));key.fingerprint = mix(key.fingerprint, valueHash(root)); }
    key.pin();
    return nb::make_tuple(leafObjects, nb::cast(std::move(key)));
  }
 private:
  struct Selection { std::vector<Id> required;Id slice = 0; };
  std::shared_ptr<RuntimeArena> arena;
  nb::object engine;
  std::vector<Id> requested, leaves, ordered;
  std::unordered_map<Id, Selection> selections;
  std::unordered_map<Id, Id> leafIndices, expressionIndices;
  std::unordered_map<Id, std::pair<std::size_t, bool>> hashes;
  Binding binding(Id value) const {
    value = arena->source(value);
    const auto node = arena->values[value];
    return node.storage ? Binding{0, leafIndices.at(value), 0} : Binding{1, expressionIndices.at(node.origin), node.port};
  }
  std::size_t valueHash(Id value) const {
    value = arena->source(value);
    const auto node = arena->values[value];
    return node.storage ? mix(1, node.type) : mix(hashes.at(node.origin).first, node.port);
  }
  Id carrier() const {
    Id preferred = arena->values[requested.front()].type, fallback = 0;
    std::vector<Id> pending(requested.rbegin(), requested.rend());
    std::unordered_set<Id> seen;
    while (!pending.empty()) {
      Id id = arena->source(pending.back());pending.pop_back();
      if (!seen.insert(id).second) continue;
      const auto value = arena->values[id];
      if (value.storage) { if (value.type == preferred) return id;if (!fallback) fallback = id; }
      else {
        const auto expression = arena->expressions[value.origin];
        for (Id index = expression.count; index; --index) pending.push_back(arena->edges[expression.edge + index - 1]);
      }
    }
    if (!fallback) throw nb::value_error("a constant-only deferred graph needs a materialized input carrier");
    return fallback;
  }
};

int visitType(PyObject *self, visitproc visit, void *arg) {
  return visit(reinterpret_cast<PyObject *>(Py_TYPE(self)), arg);
}
int arenaTraverse(PyObject *self, visitproc visit, void *arg) {
  if (nb::inst_ready(self)) if (int result = nb::inst_ptr<ArenaOwner>(self)->arena->traverse(visit, arg)) return result;
  return visitType(self, visit, arg);
}
int arenaClear(PyObject *self) {
  if (nb::inst_ready(self)) nb::inst_ptr<ArenaOwner>(self)->arena->clear();
  return 0;
}
template<class T> int ownerTraverse(PyObject *self, visitproc visit, void *arg) {
  if (nb::inst_ready(self)) {
    const auto &owner = nb::inst_ptr<T>(self)->engine;
    if (owner) if (int result = visit(owner.ptr(), arg)) return result;
  }
  return visitType(self, visit, arg);
}
template<class T> int ownerClear(PyObject *self) {
  if (nb::inst_ready(self)) nb::inst_ptr<T>(self)->engine = {};
  return 0;
}
int keyTraverse(PyObject *self, visitproc visit, void *arg) {
  if (nb::inst_ready(self)) {
    const auto &owner = nb::inst_ptr<PlanKey>(self)->owner;
    if (owner) if (int result = visit(owner.ptr(), arg)) return result;
  }
  return visitType(self, visit, arg);
}
int keyClear(PyObject *self) {
  if (nb::inst_ready(self)) nb::inst_ptr<PlanKey>(self)->owner = {};
  return 0;
}
template<class T> const PyType_Slot *ownerSlots() {
  static const PyType_Slot slots[] = {{Py_tp_traverse, reinterpret_cast<void *>(ownerTraverse<T>)},
                                    {Py_tp_clear, reinterpret_cast<void *>(ownerClear<T>)}, {0, nullptr}};
  return slots;
}
} // namespace

TensorHandle::TensorHandle(std::shared_ptr<RuntimeArena> owner, Id id, nb::object engine)
    : arena(std::move(owner)), value(id), engine(std::move(engine)) { arena->retain(value); }
TensorHandle::TensorHandle(const TensorHandle &other) : TensorHandle(other.arena, other.value, other.engine) {}
TensorHandle::TensorHandle(TensorHandle &&other) noexcept
    : arena(std::move(other.arena)), value(std::exchange(other.value, 0)), engine(std::move(other.engine)) {}
TensorHandle::~TensorHandle() { if (arena) arena->release(value); }
nb::tuple TensorHandle::shape() const { return arena->types[typeId()].shape; }
nb::object TensorHandle::dtype() const { return arena->types[typeId()].dtype; }
nb::object TensorHandle::storage() const {
  auto id = arena->values[value].storage;
  return id ? arena->storages[id] : nb::none();
}
Id TensorHandle::typeId() const { return arena->values[value].type; }
const std::vector<std::int64_t> &TensorHandle::dimensions() const { return arena->types[typeId()].dimensions; }

nb::tuple bindPending(nb::handle engine, nb::handle tensorType, nb::handle metadata,
                      nb::tuple inputs, nb::tuple dependencies) {
  EngineLock lock(engine);
  engine.attr("_check_open")();
  auto arena = arenaOf(engine);
  if (!arena->tensorType) arena->tensorType = nb::borrow<nb::object>(tensorType);
  const auto metadataId = arena->metadata(metadata);
  std::vector<Id> values;
  for (nb::handle input : inputs) {
    auto &tensor = nb::cast<TensorHandle &>(input);
    if (tensor.arena != arena) throw nb::value_error("operation inputs must belong to the same Engine");
    values.push_back(tensor.value);
  }
  const auto id = arena->expression(metadataId, values, arena->dependency(dependencies));
  // A construction pin also covers templates with no returned ports.
  arena->retainExpression(id);
  nb::list result;
  try {
    const auto types = arena->templates[metadataId].outputs;
    for (Id port = 0; port < types.size(); ++port) {
      Id value = arena->value(types[port], id, port);
      result.append(arena->wrap(value, engine, tensorType));
    }
  } catch (...) { arena->release(id, true);throw; }
  arena->release(id, true);
  return nb::tuple(result);
}
} // namespace fuga::lazy

void bindLazy(nb::module_ &module) {
  using namespace fuga::lazy;
  static const PyType_Slot arenaSlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(arenaTraverse)},
                                         {Py_tp_clear, reinterpret_cast<void *>(arenaClear)}, {0, nullptr}};
  static const PyType_Slot keySlots[] = {{Py_tp_traverse, reinterpret_cast<void *>(keyTraverse)},
                                       {Py_tp_clear, reinterpret_cast<void *>(keyClear)}, {0, nullptr}};
  nb::class_<ArenaOwner>(module, "RuntimeArena", nb::type_slots(arenaSlots))
      .def(nb::init<>()).def_prop_ro("stats", [](const ArenaOwner &self) { return self.arena->stats(); });
  nb::class_<TensorHandle>(module, "TensorHandle", nb::is_weak_referenceable(), nb::type_slots(ownerSlots<TensorHandle>()))
      .def("__init__", [](TensorHandle *self, nb::handle engine, nb::handle storage) {
        auto arena = arenaOf(engine);
        auto dtype = nb::module_::import_("fugatensor.dtypes").attr("DType")(storage.attr("dtype"));
        Id type = arena->type(storage.attr("shape"), dtype);
        Id value = arena->value(type, 0, 0, 0, storage);
        new (self) TensorHandle(std::move(arena), value, nb::borrow<nb::object>(engine));
      })
      .def_prop_ro("_engine", [](const TensorHandle &self) { return self.engine; })
      .def_prop_ro("_shape", &TensorHandle::shape).def_prop_ro("_dtype", &TensorHandle::dtype)
      .def_prop_rw("_storage", &TensorHandle::storage,
          [](TensorHandle &self, nb::handle storage) { self.arena->publish(self.value, storage); })
      .def_prop_ro("_identity", [](const TensorHandle &self) { return ValueIdentity(self.arena, self.engine, self.value, false); })
      .def_prop_ro("_origin", [](const TensorHandle &self) -> nb::object {
        const auto value = self.arena->values[self.value];
        return value.origin ? nb::cast(ResultView(self.arena, self.engine, value.origin, value.port)) : nb::none();
      })
      .def_prop_ro("_pending", [](const TensorHandle &self) -> nb::object {
        const auto value = self.arena->values[self.value];
        if (value.storage) return nb::none();
        if (value.alias) return nb::cast(AliasView(self.arena, self.engine, value.alias, false));
        return nb::cast(ResultView(self.arena, self.engine, value.origin, value.port));
      });
  nb::class_<ValueIdentity>(module, "ValueIdentity", nb::type_slots(ownerSlots<ValueIdentity>()))
      .def("__hash__", &ValueIdentity::hash)
      .def("__eq__", [](const ValueIdentity &self, const ValueIdentity &other) { return self.same(other); }, nb::is_operator());
  nb::class_<ExpressionView>(module, "Expression", nb::type_slots(ownerSlots<ExpressionView>()))
      .def_prop_ro("module", &ExpressionView::module).def_prop_ro("inputs", &ExpressionView::inputs)
      .def_prop_ro("dependencies", &ExpressionView::dependencies)
      .def("__hash__", &ExpressionView::hash)
      .def("__eq__", [](const ExpressionView &self, const ExpressionView &other) { return self.same(other); }, nb::is_operator());
  nb::class_<ResultView>(module, "Result", nb::type_slots(ownerSlots<ResultView>()))
      .def(nb::init<const ExpressionView &, Id>()).def_prop_ro("expression", &ResultView::owner)
      .def_ro("index", &ResultView::port).def("__hash__", &ResultView::hash)
      .def("__eq__", &ResultView::same, nb::is_operator());
  nb::class_<AliasView>(module, "Alias", nb::type_slots(ownerSlots<AliasView>()))
      .def_prop_ro("source", &AliasView::source);
  nb::class_<PlanKey>(module, "PlanKey", nb::type_slots(keySlots))
      .def("__hash__", [](const PlanKey &self) { return pythonHash(self.fingerprint); })
      .def("__eq__", &PlanKey::same, nb::is_operator());
  module.def("bind_pending", &bindPending);
  module.def("detach_tensor", [](nb::handle object, nb::handle tensorType) {
    auto &self = nb::cast<TensorHandle &>(object);
    auto &arena = *self.arena;
    auto storage = self.storage();
    Id value = arena.value(self.typeId(), 0, 0, storage.is_none() ? self.value : 0,
                           storage.is_none() ? nb::handle() : nb::handle(storage));
    return arena.wrap(value, self.engine, tensorType);
  });
  module.def("source_tensor", [](nb::handle object) {
    auto &self = nb::cast<TensorHandle &>(object);
    Id source = self.arena->source(self.value);
    return source == self.value ? nb::borrow<nb::object>(object) : self.arena->wrap(source, self.engine);
  });
  module.def("pending_roots", [](nb::sequence tensors) {
    nb::list roots;
    std::unordered_set<Id> seen;
    for (nb::handle object : tensors) {
      auto &self = nb::cast<TensorHandle &>(object);
      Id source = self.arena->source(self.value);
      if (!self.arena->values[source].storage && seen.insert(source).second)
        roots.append(source == self.value ? nb::borrow<nb::object>(object) : self.arena->wrap(source, self.engine));
    }
    return roots;
  });
  module.def("publish_pending", [](nb::sequence roots, nb::sequence storages) {
    if (nb::len(roots) != nb::len(storages)) throw std::runtime_error("ASK returned an unexpected number of graph results");
    for (std::size_t index = 0; index < nb::len(roots); ++index) {
      auto &root = nb::cast<TensorHandle &>(roots[index]);
      root.arena->publish(root.value, storages[index]);
    }
  });
  module.def("collect_pending", [](nb::sequence roots) {
    if (!nb::len(roots)) throw nb::value_error("pending roots must be nonempty");
    auto &root = nb::cast<TensorHandle &>(roots[0]);
    return Collection(root.arena, root.engine).collect(roots, false);
  });
  module.def("describe_pending", [](nb::sequence roots) {
    auto &root = nb::cast<TensorHandle &>(roots[0]);
    return Collection(root.arena, root.engine).collect(roots, true);
  });
}
