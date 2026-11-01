#include "graph.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
std::size_t integer(PyObject *value, const std::string &name) {
  if (!PyLong_CheckExact(value))
    throw std::invalid_argument(name + " must be an ordinary nonnegative integer");
  unsigned long long result = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::invalid_argument(name + " must fit the nonnegative native index range");
  }
  if (result > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(name + " exceeds the native index range");
  return static_cast<std::size_t>(result);
}

Py_ssize_t length(PyObject *value, const std::string &name) {
  if (PyList_Check(value)) return PyList_GET_SIZE(value);
  if (PyTuple_Check(value)) return PyTuple_GET_SIZE(value);
  throw std::invalid_argument(name + " must be a list or tuple");
}

PyObject *item(PyObject *value, Py_ssize_t index) {
  return PyList_Check(value) ? PyList_GET_ITEM(value, index) : PyTuple_GET_ITEM(value, index);
}

std::vector<std::size_t> indices(PyObject *value, const std::string &name) {
  const auto count = length(value, name);
  std::vector<std::size_t> result;
  result.reserve(count);
  for (Py_ssize_t index = 0; index < count; ++index)
    result.push_back(integer(item(value, index), name + "[" + std::to_string(index) + "]"));
  return result;
}

nb::dict analyze(nb::handle inputCount, nb::handle operands, nb::handle resultCounts,
                 nb::handle outputs, nb::handle selectedInputs, nb::handle effects) {
  // Snapshot the whole Python request while holding the GIL. The analysis
  // below touches only C++ values, so callers may mutate their original lists
  // or run another analysis concurrently without racing this request.
  fuga::Graph graph;
  graph.inputCount = integer(inputCount.ptr(), "input_count");
  const auto operations = length(operands.ptr(), "operands");
  graph.operands.reserve(operations);
  for (Py_ssize_t index = 0; index < operations; ++index)
    graph.operands.push_back(indices(item(operands.ptr(), index), "operands[" + std::to_string(index) + "]"));
  graph.resultCounts = indices(resultCounts.ptr(), "result_counts");
  graph.outputs = indices(outputs.ptr(), "outputs");
  graph.selectedInputs = indices(selectedInputs.ptr(), "selected_inputs");
  const auto effectCount = length(effects.ptr(), "effects");
  graph.effects.reserve(effectCount);
  for (Py_ssize_t index = 0; index < effectCount; ++index) {
    PyObject *value = item(effects.ptr(), index);
    if (!PyBool_Check(value)) throw std::invalid_argument("effects must contain Python bool flags");
    graph.effects.push_back(value == Py_True ? 1 : 0);
  }

  fuga::Analysis analysis;
  {
    nb::gil_scoped_release release;
    analysis = fuga::analyze(graph);
  }
  nb::list active;
  for (std::uint8_t flag : analysis.activeValues) active.append(flag != 0);
  nb::dict result;
  result["active_values"] = active;
  result["reverse_operations"] = nb::cast(analysis.reverseOperations);
  result["live_operations"] = nb::cast(analysis.liveOperations);
  result["value_count"] = analysis.valueCount;
  result["operation_count"] = analysis.operationCount;
  return result;
}
} // namespace

NB_MODULE(_native, module) {
  module.doc() = "Fuga's hardware-independent SSA dependency and liveness planner";
  module.attr("__version__") = FUGA_VERSION;
  module.attr("graph_api_version") = 1;
  module.def("analyze", &analyze, "input_count"_a, "operands"_a, "result_counts"_a,
             "outputs"_a, "selected_inputs"_a, "effects"_a,
             "Analyze a topologically ordered SSA graph in C++, releasing the GIL.\n\n"
             "Values are inputs followed by each operation's results. active_values\n"
             "tracks dependence on selected inputs. reverse_operations is the reverse\n"
             "schedule needed by live active outputs. live_operations is the forward\n"
             "DCE schedule including effectful work and inactive primal coefficients.\n"
             "Multi-result operations conservatively depend on all their operands.\n"
             "Malformed graphs and invalid index types raise ValueError.");
}
