#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fuga {

// Values are numbered inputs first, then each operation's result range.
// Every operand must refer to an input or an earlier operation result.
struct Graph {
  std::size_t inputCount = 0;
  std::vector<std::vector<std::size_t>> operands;
  std::vector<std::size_t> resultCounts;
  std::vector<std::size_t> outputs;
  std::vector<std::size_t> selectedInputs;
  std::vector<std::uint8_t> effects;
};

struct Analysis {
  std::vector<std::uint8_t> activeValues;
  std::vector<std::size_t> reverseOperations;
  std::vector<std::size_t> liveOperations;
  std::size_t valueCount = 0;
  std::size_t operationCount = 0;
};

// Throws std::invalid_argument for malformed graphs. No Python, allocation of
// tensor data, target selection, or numerical differentiation happens here.
// Multi-result nodes conservatively depend on all their operands; Python's
// derivative-rule dispatcher supplies primitive or custom VJP semantics.
Analysis analyze(const Graph &graph);

} // namespace fuga
