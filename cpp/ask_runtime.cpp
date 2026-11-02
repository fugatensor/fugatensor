#include "ask/Runtime/CUDA.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ask::runtime;

namespace {
struct BufferView {
  Py_buffer view{};
  BufferView(nb::handle object,bool writable) {
    if(PyObject_GetBuffer(object.ptr(),&view,writable?PyBUF_CONTIG:PyBUF_CONTIG_RO)<0)
      throw nb::python_error();
  }
  ~BufferView() { PyBuffer_Release(&view); }
};
}

void bindAskRuntime(nb::module_ &module) {
  nb::exception<CUDAError>(module, "CUDAError", nb::module_::import_("fugatensor._ask.cuda").attr("CUDAError"));
  nb::class_<DeviceBuffer>(module,"CUDADeviceBuffer")
      .def_prop_ro("data_ptr",&DeviceBuffer::pointer)
      .def_prop_ro("nbytes",&DeviceBuffer::size)
      .def_prop_ro("readonly",&DeviceBuffer::readonly)
      .def("copy_from_host",[](DeviceBuffer &buffer,nb::handle data){
        BufferView source(data,false);
        nb::gil_scoped_release release;buffer.copyFromHost(source.view.buf,source.view.len);
      })
      .def("copy_to_host",[](const DeviceBuffer &buffer,nb::handle data){
        BufferView destination(data,true);
        nb::gil_scoped_release release;buffer.copyToHost(destination.view.buf,destination.view.len);
      })
      .def("to_host_bytes",[](const DeviceBuffer &buffer){
        std::string data;{nb::gil_scoped_release release;data=buffer.copyToHost();}
        return nb::bytes(data.data(),data.size());
      });
  nb::class_<Kernel>(module,"CUDAKernel")
      .def_prop_ro("entry",&Kernel::entry)
      .def_prop_ro("max_threads",&Kernel::maxThreads);
  nb::class_<Event>(module,"CUDAEvent")
      .def("query",&Event::query,nb::call_guard<nb::gil_scoped_release>())
      .def("wait",&Event::wait,nb::call_guard<nb::gil_scoped_release>());
  nb::class_<PreparedProgram>(module,"CUDAProgram")
      .def("submit",&PreparedProgram::submit,nb::arg("inputs"),nb::arg("outputs"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("run",&PreparedProgram::run,nb::arg("inputs"),nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Session>(module,"CUDASession")
      .def_static("create",&Session::create,nb::arg("device")=0,nb::call_guard<nb::gil_scoped_release>())
      .def("allocate",&Session::allocate,nb::arg("nbytes"),nb::call_guard<nb::gil_scoped_release>())
      .def("import_buffer",&Session::importBuffer,nb::arg("pointer"),nb::arg("nbytes"),
           nb::arg("readonly")=false,nb::call_guard<nb::gil_scoped_release>())
      .def("load",[](Session &session,nb::bytes data,const std::string &entry,
                      const std::vector<std::string> &types,int arch,const std::vector<unsigned> &widths){
        std::string binary(static_cast<const char *>(data.data()),data.size());
        nb::gil_scoped_release release;return session.load(binary,entry,types,arch,widths);
      },nb::arg("cubin"),nb::arg("entry"),nb::arg("required_types"),nb::arg("arch"),
        nb::arg("parameter_widths"))
      .def("submit",[](Session &session,const std::shared_ptr<Kernel> &kernel,
                         nb::list bindings,std::array<unsigned,3> grid,
                         std::array<unsigned,3> block,unsigned sharedBytes){
        std::vector<KernelArgument> arguments;
        for(nb::handle binding:bindings) {
          KernelArgument argument;
          if(nb::isinstance<nb::bytes>(binding)) {
            auto bytes=nb::cast<nb::bytes>(binding);
            argument.scalar.assign(static_cast<const char *>(bytes.data()),bytes.size());
            if(argument.scalar.empty()) throw CUDAError("scalar argument cannot be empty");
          } else if(!binding.is_none()) argument.buffer=nb::cast<std::shared_ptr<DeviceBuffer>>(binding);
          arguments.push_back(std::move(argument));
        }
        nb::gil_scoped_release release;return session.submit(kernel,arguments,grid,block,sharedBytes);
      },nb::arg("kernel"),nb::arg("bindings"),nb::arg("grid"),nb::arg("block"),nb::arg("shared_bytes")=0)
      .def("prepare_program",[](Session &session,std::vector<std::size_t> sizes,
          std::vector<std::size_t> inputs,std::vector<std::size_t> outputs,
          nb::list definitions,bool graph) {
        std::vector<PlanStep> steps;
        for(nb::handle definition:definitions) {
          auto item=nb::cast<nb::dict>(definition);PlanStep step;
          step.kernel=nb::cast<std::shared_ptr<Kernel>>(item["kernel"]);
          step.grid=nb::cast<std::array<unsigned,3>>(item["grid"]);
          step.block=nb::cast<std::array<unsigned,3>>(item["block"]);
          step.sharedBytes=nb::cast<unsigned>(item["shared_bytes"]);
          step.inputs=nb::cast<std::vector<std::size_t>>(item["inputs"]);
          step.outputs=nb::cast<std::vector<std::size_t>>(item["outputs"]);
          for(nb::handle binding:nb::cast<nb::list>(item["arguments"])) {
            PlanArgument argument;
            if(nb::isinstance<nb::bytes>(binding)) {
              auto scalar=nb::cast<nb::bytes>(binding);
              argument.scalar.assign(static_cast<const char *>(scalar.data()),scalar.size());
              if(argument.scalar.empty()) throw CUDAError("scalar argument cannot be empty");
            } else if(!binding.is_none()) argument.buffer=nb::cast<std::int64_t>(binding);
            step.arguments.push_back(std::move(argument));
          }
          steps.push_back(std::move(step));
        }
        nb::gil_scoped_release release;
        return session.prepareProgram(std::move(sizes),std::move(inputs),std::move(outputs),std::move(steps),graph);
      },nb::arg("buffer_sizes"),nb::arg("inputs"),nb::arg("outputs"),nb::arg("steps"),nb::arg("graph")=false)
      .def("wait_for_external_stream",&Session::waitForExternalStream,nb::call_guard<nb::gil_scoped_release>())
      .def("check_types",&Session::checkTypes,nb::call_guard<nb::gil_scoped_release>())
      .def("synchronize",&Session::synchronize,nb::call_guard<nb::gil_scoped_release>())
      .def("reap",&Session::reap,nb::call_guard<nb::gil_scoped_release>())
      .def("close",&Session::close,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("closed",&Session::closed,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("stream",&Session::stream,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("device",&Session::device)
      .def_prop_ro("capability",&Session::capability)
      .def_prop_ro("device_name",&Session::deviceName)
      .def_prop_ro("stats",&Session::stats,nb::call_guard<nb::gil_scoped_release>());
}
