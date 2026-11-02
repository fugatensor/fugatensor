#include "ask/Frontend/CompilerAPI.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace ask;

void bindCompiler(nb::module_ &module) {
  module.def("compiler_library_path", &compilerLibraryPath,
             nb::call_guard<nb::gil_scoped_release>());
  module.def("capabilities", &compilerCapabilitiesJSON,
             nb::call_guard<nb::gil_scoped_release>());
  module.def("prepare", &prepareJSON, nb::arg("source"), nb::arg("optimize") = false,
     nb::arg("fold_constants") = true, nb::arg("eliminate_dead_code") = true,
     nb::arg("fuse_elementwise") = true, nb::call_guard<nb::gil_scoped_release>());
  module.def("partition", &partitionJSON, nb::arg("source"), nb::arg("optimize") = true,
     nb::arg("fold_constants") = true, nb::arg("eliminate_dead_code") = true,
     nb::arg("fuse_elementwise") = true,
     nb::call_guard<nb::gil_scoped_release>());
  module.def("codegen", &codegenJSON, nb::arg("source"), nb::arg("target"),
     nb::arg("output") = "", nb::arg("arch") = 0, nb::arg("block_size") = 128,
     nb::arg("ptx_version") = 80, nb::arg("optimize") = true,
     nb::call_guard<nb::gil_scoped_release>());
}
