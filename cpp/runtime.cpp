#include "ask/Runtime/CUDA.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Scheduling and ownership live in ASK. This bridge converts the input list
// once and calls the shared C++ runtime without a Python ASK execution call.
void bindExecution(nb::module_ &module) {
  module.def("run_program", [](const std::shared_ptr<ask::runtime::PreparedProgram> &program,
      const std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> &inputs) {
    if (!program) throw std::invalid_argument("program cannot be null");
    return program->run(inputs);
  }, nb::arg("program"), nb::arg("inputs"), nb::call_guard<nb::gil_scoped_release>());
}
