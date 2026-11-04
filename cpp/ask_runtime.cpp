#include "ask/Runtime/CUDA.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <unordered_map>

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

// Only freshly constructed tuples of integers and strings are retained here.
// They cannot refer back to this descriptor, a session or a Python tensor.
struct TensorPort {
  nb::tuple shape;
  nb::str dtype;
};

class ProgramPorts {
public:
  ProgramPorts(nb::sequence inputs, nb::sequence outputs)
      : inputs(readPorts(inputs)), outputs(readPorts(outputs)) {}

  nb::list run(PreparedProgram &program, nb::handle session,
               nb::handle values, nb::handle tensorType) const {
    auto tensors = checkPorts(values, inputs, "inputs", tensorType);
    prepareAccess(tensors, session);
    session.attr("reap")();
    auto bound = buffers(tensors);
    std::pair<std::vector<std::shared_ptr<DeviceBuffer>>, std::shared_ptr<Event>> result;
    {
      nb::gil_scoped_release release;
      result = program.run(bound);
    }
    session.attr("_retain")(nb::cast(result.second), tensors);
    std::unordered_map<DeviceBuffer *, nb::object> handles;
    handles.reserve(bound.size() + result.first.size());
    for (std::size_t index = 0; index < bound.size(); ++index)
      handles.insert_or_assign(bound[index].get(), nb::borrow<nb::object>(tensors[index]));
    nb::list wrapped;
    for (std::size_t index = 0; index < result.first.size(); ++index) {
      const auto &buffer = result.first[index];
      auto found = handles.find(buffer.get());
      if (found == handles.end()) {
        const auto &port = outputs[index];
        found = handles.emplace(buffer.get(), tensorType(
            session, nb::cast(buffer), port.shape, port.dtype)).first;
      }
      wrapped.append(found->second);
    }
    return wrapped;
  }

  nb::object submit(PreparedProgram &program, nb::handle session,
                    nb::handle inputValues, nb::handle outputValues,
                    nb::handle tensorType) const {
    auto inputTensors = checkPorts(inputValues, inputs, "inputs", tensorType);
    auto outputTensors = checkPorts(outputValues, outputs, "outputs", tensorType);
    prepareAccess(inputTensors, session);
    prepareAccess(outputTensors, session);
    session.attr("reap")();
    auto inputBuffers = buffers(inputTensors), outputBuffers = buffers(outputTensors);
    std::shared_ptr<Event> event;
    {
      nb::gil_scoped_release release;
      event = program.submit(inputBuffers, outputBuffers);
    }
    auto references = nb::steal<nb::object>(PySequence_Concat(inputTensors.ptr(), outputTensors.ptr()));
    if (!references.is_valid()) throw nb::python_error();
    return session.attr("_retain")(nb::cast(event), references);
  }

private:
  std::vector<TensorPort> inputs, outputs;

  static std::vector<TensorPort> readPorts(nb::sequence definitions) {
    std::vector<TensorPort> result;
    result.reserve(nb::len(definitions));
    for (nb::handle definition : definitions) {
      auto port = nb::cast<nb::dict>(definition);
      const auto shape = nb::cast<std::vector<std::int64_t>>(port["shape"]);
      const auto dtype = nb::cast<std::string>(port["element_type"]);
      result.push_back({nb::tuple(nb::cast(shape)), nb::str(dtype.c_str())});
    }
    return result;
  }

  static nb::tuple checkPorts(nb::handle values, const std::vector<TensorPort> &ports,
                             const char *kind, nb::handle tensorType) {
    if (!PyTuple_Check(values.ptr()) && !PyList_Check(values.ptr()))
      throw CUDAError("program ports must be a list or tuple");
    // Freeze mutable caller lists before any callback can release the GIL.
    nb::tuple tensors(values);
    if (tensors.size() != ports.size())
      throw CUDAError("program expects " + std::to_string(ports.size()) + " " + kind);
    for (std::size_t index = 0; index < ports.size(); ++index) {
      nb::handle tensor = tensors[index];
      if (!nb::isinstance(tensor, tensorType))
        throw CUDAError("program ports require DeviceTensor values");
      if (!nb::tuple(tensor.attr("shape")).equal(ports[index].shape) ||
          !tensor.attr("dtype").equal(ports[index].dtype))
        throw CUDAError("program port shape/dtype differs from its compiled type");
    }
    return tensors;
  }

  static void prepareAccess(nb::tuple tensors, nb::handle session) {
    for (nb::handle tensor : tensors) tensor.attr("_prepare_access")(session);
  }

  static std::vector<std::shared_ptr<DeviceBuffer>> buffers(nb::tuple tensors) {
    std::vector<std::shared_ptr<DeviceBuffer>> result;
    result.reserve(tensors.size());
    for (nb::handle tensor : tensors)
      result.push_back(nb::cast<std::shared_ptr<DeviceBuffer>>(tensor.attr("_buffer")));
    return result;
  }
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
  nb::class_<ProgramPorts>(module,"CUDAProgramPorts")
      .def(nb::init<nb::sequence, nb::sequence>(), nb::arg("inputs"), nb::arg("outputs"))
      .def("run", &ProgramPorts::run, nb::arg("program"), nb::arg("session"),
           nb::arg("inputs").none(), nb::arg("tensor_type"))
      .def("submit", &ProgramPorts::submit, nb::arg("program"), nb::arg("session"),
           nb::arg("inputs").none(), nb::arg("outputs").none(), nb::arg("tensor_type"));
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
          step.inputs=nb::cast<std::vector<std::size_t>>(item["inputs"]);
          step.outputs=nb::cast<std::vector<std::size_t>>(item["outputs"]);
          if(item.contains("matmul")) {
            auto description=nb::cast<nb::dict>(item["matmul"]);
            MatmulDescription matmul;
            matmul.elementType=nb::cast<std::string>(description["element_type"]);
            matmul.m=nb::cast<std::uint64_t>(description["m"]);
            matmul.n=nb::cast<std::uint64_t>(description["n"]);
            matmul.k=nb::cast<std::uint64_t>(description["k"]);
            matmul.batchCount=nb::cast<std::uint64_t>(description["batch_count"]);
            matmul.transposeA=nb::cast<bool>(description["transpose_a"]);
            matmul.transposeB=nb::cast<bool>(description["transpose_b"]);
            step.matmul=std::move(matmul);
            steps.push_back(std::move(step));
            continue;
          }
          step.kernel=nb::cast<std::shared_ptr<Kernel>>(item["kernel"]);
          step.grid=nb::cast<std::array<unsigned,3>>(item["grid"]);
          step.block=nb::cast<std::array<unsigned,3>>(item["block"]);
          step.sharedBytes=nb::cast<unsigned>(item["shared_bytes"]);
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
