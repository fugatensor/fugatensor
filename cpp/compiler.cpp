#include "ask/Frontend/CompilerAPI.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace ask;

void bindCompiler(nb::module_ &module) {
  namespace api = ask::compiler;
  nb::enum_<api::CodegenTarget>(module, "CodegenTarget")
      .value("HostLLVM", api::CodegenTarget::HostLLVM)
      .value("HostObject", api::CodegenTarget::HostObject)
      .value("CUDALLVM", api::CodegenTarget::CUDALLVM);
  nb::class_<api::PrepareResult>(module, "PrepareResult")
      .def_ro("ir", &api::PrepareResult::ir)
      .def_ro("shape_snapshot_json", &api::PrepareResult::shapeSnapshotJSON)
      .def_ro("external_requests_json", &api::PrepareResult::externalRequestsJSON)
      .def_ro("diagnostics_json", &api::PrepareResult::diagnosticsJSON);
  nb::class_<api::PartitionResult>(module, "PartitionResult")
      .def_ro("ir", &api::PartitionResult::ir)
      .def_ro("shape_snapshot_json", &api::PartitionResult::shapeSnapshotJSON)
      .def_ro("external_requests_json", &api::PartitionResult::externalRequestsJSON)
      .def_ro("execution_plan_json", &api::PartitionResult::executionPlanJSON)
      .def_ro("step_sources", &api::PartitionResult::stepSources)
      .def_ro("diagnostics_json", &api::PartitionResult::diagnosticsJSON);
  nb::class_<api::CodegenResult>(module, "CodegenResult")
      .def_ro("target", &api::CodegenResult::target)
      .def_ro("output_path", &api::CodegenResult::outputPath)
      .def_ro("ir", &api::CodegenResult::ir)
      .def_ro("shape_snapshot_json", &api::CodegenResult::shapeSnapshotJSON)
      .def_ro("external_requests_json", &api::CodegenResult::externalRequestsJSON)
      .def_ro("abi_json", &api::CodegenResult::abiJSON)
      .def_ro("cuda_artifact_json", &api::CodegenResult::cudaArtifactJSON)
      .def_ro("diagnostics_json", &api::CodegenResult::diagnosticsJSON);
  module.def("prepare_typed", [](const std::string &source, bool optimize,
      bool foldConstants, bool eliminateDeadCode, bool fuseElementwise) {
    return api::prepare(source, {optimize, {foldConstants, eliminateDeadCode, fuseElementwise}});
  }, nb::arg("source"), nb::arg("optimize") = false, nb::arg("fold_constants") = true,
     nb::arg("eliminate_dead_code") = true, nb::arg("fuse_elementwise") = true,
     nb::call_guard<nb::gil_scoped_release>());
  module.def("partition_typed", [](const std::string &source, bool optimize) {
    api::PartitionOptions options;
    options.optimize = optimize;
    return api::partition(source, options);
  }, nb::arg("source"), nb::arg("optimize") = true, nb::call_guard<nb::gil_scoped_release>());
  module.def("codegen_typed", [](const std::string &source, api::CodegenTarget target,
      const std::string &output, unsigned architecture, unsigned blockSize,
      unsigned ptxVersion, bool optimize) {
    api::CodegenOptions options;
    options.target = target;
    options.outputPath = output;
    options.architecture = architecture;
    options.blockSize = blockSize;
    options.ptxVersion = ptxVersion;
    options.optimize = optimize;
    return api::codegen(source, options);
  }, nb::arg("source"), nb::arg("target"), nb::arg("output") = "", nb::arg("arch") = 0,
     nb::arg("block_size") = 128, nb::arg("ptx_version") = 80, nb::arg("optimize") = true,
     nb::call_guard<nb::gil_scoped_release>());
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
