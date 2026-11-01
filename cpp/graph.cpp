#include "graph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace fuga {
namespace {
[[noreturn]] void invalid(const std::string &message) {
  throw std::invalid_argument(message);
}
} // namespace

Analysis analyze(const Graph &graph) {
  const std::size_t operations = graph.operands.size();
  if (graph.resultCounts.size() != operations || graph.effects.size() != operations)
    invalid("operands, result_counts and effects must have the same operation count");
  const std::size_t maximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (graph.inputCount > maximum)
    invalid("input_count exceeds the supported index range");

  std::vector<std::size_t> starts;
  starts.reserve(operations + 1);
  std::size_t values = graph.inputCount;
  for (std::size_t index = 0; index < operations; ++index) {
    starts.push_back(values);
    for (std::size_t operand : graph.operands[index])
      if (operand >= values)
        invalid("operation " + std::to_string(index) + " operand " +
                std::to_string(operand) + " must refer to a previously defined value");
    if (graph.resultCounts[index] > maximum - values)
      invalid("result counts overflow the supported value index range");
    if (graph.effects[index] > 1)
      invalid("effect flags must be boolean");
    values += graph.resultCounts[index];
  }
  starts.push_back(values);
  for (std::size_t output : graph.outputs)
    if (output >= values)
      invalid("output index is outside the defined value range");
  std::unordered_set<std::size_t> selected;
  for (std::size_t input : graph.selectedInputs) {
    if (input >= graph.inputCount)
      invalid("selected_inputs must refer to graph inputs");
    if (!selected.insert(input).second)
      invalid("selected_inputs must not contain duplicates");
  }

  Analysis result;
  result.valueCount = values;
  result.operationCount = operations;
  result.activeValues.assign(values, 0);
  for (std::size_t input : graph.selectedInputs)
    result.activeValues[input] = 1;
  for (std::size_t index = 0; index < operations; ++index) {
    const bool active = std::any_of(graph.operands[index].begin(), graph.operands[index].end(),
                                   [&](std::size_t value) { return result.activeValues[value] != 0; });
    std::fill(result.activeValues.begin() + starts[index], result.activeValues.begin() + starts[index + 1],
              static_cast<std::uint8_t>(active));
  }

  // Differentiation reachability differs from ordinary liveness. For x * f(y)
  // differentiated only with respect to x, f(y) is a live coefficient, but its
  // own VJP is not required. This distinction also handles custom call nodes.
  std::vector<std::uint8_t> backward(values, 0), liveValues(values, 0), live(operations, 0);
  for (std::size_t output : graph.outputs) {
    backward[output] = result.activeValues[output];
    liveValues[output] = 1;
  }
  for (std::size_t index = operations; index-- > 0;) {
    bool needsReverse = false;
    bool needsPrimal = graph.effects[index] != 0;
    for (std::size_t value = starts[index]; value < starts[index + 1]; ++value) {
      needsReverse = needsReverse || (backward[value] && result.activeValues[value]);
      needsPrimal = needsPrimal || liveValues[value];
    }
    if (needsReverse) {
      result.reverseOperations.push_back(index);
      for (std::size_t operand : graph.operands[index])
        if (result.activeValues[operand])
          backward[operand] = 1;
    }
    if (needsPrimal) {
      live[index] = 1;
      for (std::size_t operand : graph.operands[index])
        liveValues[operand] = 1;
    }
  }
  for (std::size_t index = 0; index < operations; ++index)
    if (live[index])
      result.liveOperations.push_back(index);
  return result;
}
} // namespace fuga
