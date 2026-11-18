#include "ask_runtime.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <new>
#include <unordered_map>
#include <utility>

namespace fuga::ask_rt {
using namespace ask::runtime;

std::size_t elemByteWidth(const std::string &dtype) {
  if (dtype == "f32" || dtype == "i32") return 4;
  if (dtype == "f16" || dtype == "bf16") return 2;
  if (dtype == "i64") return 8;
  if (dtype == "bool") return 1;
  throw CUDAError("unsupported tensor dtype: " + dtype);
}

nb::object NativeDeviceTensor::makeSubclass(
    nb::handle tensorType, nb::handle sessionRef,
    std::shared_ptr<DeviceBuffer> buffer, nb::tuple shapeTuple,
    nb::str dtypePyStr, std::vector<std::int64_t> shapeVec,
    std::string dtypeStr, std::uintptr_t sessionPtr,
    nb::handle producerStream) {
  static auto nativeBaseType = nb::type<NativeDeviceTensor>();
  if (nativeBaseType.is_valid() &&
      PyType_Check(tensorType.ptr()) &&
      PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(tensorType.ptr()),
                       reinterpret_cast<PyTypeObject *>(nativeBaseType.ptr()))) {
    nb::object obj = nb::inst_alloc(tensorType);
    auto *nd = nb::inst_ptr<NativeDeviceTensor>(obj);
    new (nd) NativeDeviceTensor();
    nd->nativeBuffer = std::move(buffer);
    nd->shapeVec = std::move(shapeVec);
    nd->dtypeStr = std::move(dtypeStr);
    nd->sessionPtr = sessionPtr;
    nd->hasOwner = false;
    nd->shapeTuple = std::move(shapeTuple);
    nd->dtypePyStr = std::move(dtypePyStr);
    nd->sessionRef = nb::borrow<nb::object>(sessionRef);
    nd->owner = nb::none();
    nd->snapshot = nb::none();
    nd->producerStream = producerStream.is_valid() ? nb::borrow<nb::object>(producerStream) : nb::none();
    nb::inst_mark_ready(obj);
    return obj;
  }
  return tensorType(nb::handle(), nb::cast(std::move(buffer)), shapeTuple, dtypePyStr);
}

CUDASessionQueue::CUDASessionQueue(std::shared_ptr<Session> sessionIn,
                                   nb::object weakRef,
                                   nb::object descriptorStreamIn)
    : session(std::move(sessionIn)),
      sessionWeakRef(std::move(weakRef)),
      descriptorStream(std::move(descriptorStreamIn)),
      sessionPtr(reinterpret_cast<std::uintptr_t>(session.get())) {
  if (session && !session->closed()) {
    try {
      cudaCtx = cuda_drv::streamContext(session->stream());
    } catch (...) {
      cudaCtx = nullptr;
    }
  }
}

CUDASessionQueue::~CUDASessionQueue() {
  pending.clear();
  if (!eventPool.empty() && cudaCtx) {
    try {
      const auto &drv = cuda_drv::driver();
      cuda_drv::ScopedCudaContext guard(cudaCtx);
      for (auto ev : eventPool) {
        if (ev) drv.cuEventDestroy(ev);
      }
    } catch (...) {
    }
  }
  eventPool.clear();
}

void CUDASessionQueue::retainAsk(std::shared_ptr<Event> event, nb::object references,
                                 std::vector<std::shared_ptr<DeviceBuffer>> buffers) {
  PendingEntry entry;
  entry.askEvent = std::move(event);
  entry.references = std::move(references);
  entry.buffers = std::move(buffers);
  pending.push_back(std::move(entry));
}

void CUDASessionQueue::retainStreamWork(nb::object references,
                                        std::vector<std::shared_ptr<DeviceBuffer>> buffers) {
  if (!session || session->closed()) return;
  const auto &drv = cuda_drv::driver();
  cuda_drv::ScopedCudaContext guard(cudaCtx);
  cuda_drv::CUevent ev = nullptr;
  if (!eventPool.empty()) {
    ev = eventPool.back();
    eventPool.pop_back();
  } else {
    auto rc = drv.cuEventCreate(&ev, cuda_drv::kCuEventDisableTiming);
    if (rc != cuda_drv::kCudaSuccess)
      throw CUDAError("cuEventCreate failed with error " + std::to_string(rc));
  }
  auto rc = drv.cuEventRecord(ev, reinterpret_cast<cuda_drv::CUstream>(session->stream()));
  if (rc != cuda_drv::kCudaSuccess) {
    eventPool.push_back(ev);
    throw CUDAError("cuEventRecord failed with error " + std::to_string(rc));
  }
  PendingEntry entry;
  entry.cuEvent = ev;
  entry.references = std::move(references);
  entry.buffers = std::move(buffers);
  pending.push_back(std::move(entry));
}

void CUDASessionQueue::reap() {
  if (!pending.empty()) {
    std::vector<PendingEntry> remaining;
    remaining.reserve(pending.size());
    bool checkedCu = false;
    const auto &drv = cuda_drv::driver();
    cuda_drv::CUcontext curCtx = nullptr;
    bool pushedCtx = false;

    for (auto &item : pending) {
      bool done = false;
      if (item.askEvent) {
        done = item.askEvent->query();
      } else if (item.cuEvent) {
        if (!checkedCu) {
          checkedCu = true;
          if (cudaCtx && drv.cuCtxGetCurrent(&curCtx) == cuda_drv::kCudaSuccess && curCtx != cudaCtx) {
            if (drv.cuCtxPushCurrent(cudaCtx) == cuda_drv::kCudaSuccess) pushedCtx = true;
          }
        }
        auto rc = drv.cuEventQuery(item.cuEvent);
        done = (rc == cuda_drv::kCudaSuccess);
      } else {
        done = true;
      }
      if (done) {
        if (item.cuEvent) eventPool.push_back(item.cuEvent);
      } else {
        remaining.push_back(std::move(item));
      }
    }
    if (pushedCtx) {
      cuda_drv::CUcontext popped = nullptr;
      drv.cuCtxPopCurrent(&popped);
    }
    pending = std::move(remaining);
  }
  if (session && !session->closed()) {
    nb::gil_scoped_release release;
    session->reap();
  }
}

void CUDASessionQueue::reapIfFull(std::size_t threshold) {
  if (pending.size() >= threshold) reap();
}

void CUDASessionQueue::clearPending() {
  for (auto &item : pending) {
    if (item.cuEvent) eventPool.push_back(item.cuEvent);
  }
  pending.clear();
  if (!eventPool.empty() && cudaCtx) {
    try {
      const auto &drv = cuda_drv::driver();
      cuda_drv::ScopedCudaContext guard(cudaCtx);
      for (auto ev : eventPool) {
        if (ev) drv.cuEventDestroy(ev);
      }
    } catch (...) {
    }
  }
  eventPool.clear();
}

CUDASessionQueue *extractSessionQueue(nb::handle session) {
  static PyObject *kRtAttr = PyUnicode_InternFromString("_rt");
  PyObject *rtObj = PyObject_GetAttr(session.ptr(), kRtAttr);
  if (!rtObj) {
    PyErr_Clear();
    return nullptr;
  }
  nb::object owned = nb::steal(rtObj);
  if (owned.is_none()) return nullptr;
  CUDASessionQueue *ptr = nullptr;
  if (nb::try_cast<CUDASessionQueue *>(owned, ptr)) return ptr;
  return nullptr;
}

namespace {

struct BufferView {
  Py_buffer view{};
  BufferView(nb::handle object, bool writable) {
    if (PyObject_GetBuffer(object.ptr(), &view, writable ? PyBUF_CONTIG : PyBUF_CONTIG_RO) < 0)
      throw nb::python_error();
  }
  ~BufferView() { PyBuffer_Release(&view); }
};

int deviceTensorTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  auto *nd = nb::inst_ptr<NativeDeviceTensor>(self);
  Py_VISIT(nd->shapeTuple.ptr());
  Py_VISIT(nd->dtypePyStr.ptr());
  Py_VISIT(nd->pyBuffer.ptr());
  Py_VISIT(nd->sessionRef.ptr());
  Py_VISIT(nd->owner.ptr());
  Py_VISIT(nd->snapshot.ptr());
  Py_VISIT(nd->producerStream.ptr());
  return 0;
}

int deviceTensorClear(PyObject *self) {
  if (!nb::inst_ready(self)) return 0;
  auto *nd = nb::inst_ptr<NativeDeviceTensor>(self);
  nd->nativeBuffer.reset();
  nd->shapeTuple.reset();
  nd->dtypePyStr.reset();
  nd->pyBuffer.reset();
  nd->sessionRef.reset();
  nd->owner.reset();
  nd->snapshot.reset();
  nd->producerStream.reset();
  return 0;
}

int sessionQueueTraverse(PyObject *self, visitproc visit, void *arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  auto *q = nb::inst_ptr<CUDASessionQueue>(self);
  Py_VISIT(q->sessionWeakRef.ptr());
  Py_VISIT(q->descriptorStream.ptr());
  for (auto &item : q->pending) {
    Py_VISIT(item.references.ptr());
  }
  return 0;
}

int sessionQueueClear(PyObject *self) {
  if (!nb::inst_ready(self)) return 0;
  auto *q = nb::inst_ptr<CUDASessionQueue>(self);
  q->sessionWeakRef.reset();
  q->descriptorStream.reset();
  q->clearPending();
  return 0;
}

}  // namespace

ProgramPorts::ProgramPorts(nb::sequence inputsIn, nb::sequence outputsIn)
    : inputs(readPorts(inputsIn)), outputs(readPorts(outputsIn)) {}

std::vector<TensorPort> ProgramPorts::readPorts(nb::sequence definitions) {
  std::vector<TensorPort> result;
  result.reserve(nb::len(definitions));
  for (nb::handle definition : definitions) {
    auto port = nb::cast<nb::dict>(definition);
    auto shapeVec = nb::cast<std::vector<std::int64_t>>(port["shape"]);
    auto dtypeStr = nb::cast<std::string>(port["element_type"]);
    result.push_back({nb::tuple(nb::cast(shapeVec)), nb::str(dtypeStr.c_str()),
                      std::move(shapeVec), std::move(dtypeStr)});
  }
  return result;
}

namespace {

bool tryFastCheckPorts(nb::handle values, const std::vector<TensorPort> &ports,
                       const char *kind, nb::handle tensorType,
                       nb::tuple &outTensors,
                       std::vector<NativeDeviceTensor *> &outNative) {
  if (!PyTuple_Check(values.ptr()) && !PyList_Check(values.ptr()))
    throw CUDAError("program ports must be a list or tuple");
  nb::tuple tensors(values);
  if (tensors.size() != ports.size())
    throw CUDAError("program expects " + std::to_string(ports.size()) + " " + kind);
  outNative.clear();
  outNative.reserve(ports.size());
  static auto nativeBaseType = nb::type<NativeDeviceTensor>();
  PyTypeObject *targetType = PyType_Check(tensorType.ptr())
                                 ? reinterpret_cast<PyTypeObject *>(tensorType.ptr())
                                 : nullptr;
  PyTypeObject *baseType = nativeBaseType.is_valid()
                               ? reinterpret_cast<PyTypeObject *>(nativeBaseType.ptr())
                               : nullptr;
  for (std::size_t index = 0; index < ports.size(); ++index) {
    nb::handle tensor = tensors[index];
    PyObject *ptr = tensor.ptr();
    if (!nb::isinstance(tensor, tensorType))
      throw CUDAError("program ports require DeviceTensor values");
    if (!baseType || !targetType || !PyObject_TypeCheck(ptr, baseType))
      return false;
    auto *nd = nb::inst_ptr<NativeDeviceTensor>(ptr);
    if (!nd->nativeBuffer)
      return false;
    if (nd->shapeVec != ports[index].shapeVec || nd->dtypeStr != ports[index].dtypeStr)
      throw CUDAError("program port shape/dtype differs from its compiled type");
    outNative.push_back(nd);
  }
  outTensors = std::move(tensors);
  return true;
}

nb::tuple slowCheckPorts(nb::handle values, const std::vector<TensorPort> &ports,
                         const char *kind, nb::handle tensorType) {
  if (!PyTuple_Check(values.ptr()) && !PyList_Check(values.ptr()))
    throw CUDAError("program ports must be a list or tuple");
  nb::tuple tensors(values);
  if (tensors.size() != ports.size())
    throw CUDAError("program expects " + std::to_string(ports.size()) + " " + kind);
  for (std::size_t index = 0; index < ports.size(); ++index) {
    nb::handle tensor = tensors[index];
    if (!nb::isinstance(tensor, tensorType))
      throw CUDAError("program ports require DeviceTensor values");
    nb::handle shape = tensor.attr("_shape");
    nb::handle dtype = tensor.attr("_dtype");
    if (!shape.equal(ports[index].shape) || !dtype.equal(ports[index].dtype))
      throw CUDAError("program port shape/dtype differs from its compiled type");
  }
  return tensors;
}

void slowPrepareAccess(nb::tuple tensors, nb::handle session) {
  for (nb::handle tensor : tensors) {
    if (!tensor.attr("_owner").is_none())
      tensor.attr("_prepare_access")(session);
  }
}

std::vector<std::shared_ptr<DeviceBuffer>> slowBuffers(nb::tuple tensors) {
  std::vector<std::shared_ptr<DeviceBuffer>> result;
  result.reserve(tensors.size());
  for (nb::handle tensor : tensors)
    result.push_back(nb::cast<std::shared_ptr<DeviceBuffer>>(tensor.attr("_buffer")));
  return result;
}

}  // namespace

void prepareAndValidateExtracted(const std::vector<NativeDeviceTensor *> &nativeTensors,
                                 nb::tuple tensors,
                                 nb::handle session,
                                 CUDASessionQueue *rt) {
  for (std::size_t i = 0; i < nativeTensors.size(); ++i) {
    auto *nd = nativeTensors[i];
    if (nd->hasOwner) {
      tensors[i].attr("_prepare_access")(session);
    } else if (nd->sessionPtr != rt->sessionPtr) {
      nb::object refSession = nd->sessionRef();
      if (refSession.is_none() || nb::cast<bool>(refSession.attr("closed")))
        throw CUDAError("device tensor's execution session is closed");
      throw CUDAError("device tensor belongs to a different execution session; "
                      "import it explicitly with from_cuda_array for cross-stream access");
    }
  }
}

bool extractInputsForSession(nb::handle values,
                             nb::handle tensorType,
                             nb::handle,
                             CUDASessionQueue *rt,
                             nb::tuple &outTensors,
                             std::vector<NativeDeviceTensor *> &outNative) {
  if (!rt || !rt->session) return false;
  if (!PyTuple_Check(values.ptr()) && !PyList_Check(values.ptr()))
    throw CUDAError("program ports must be a list or tuple");
  nb::tuple tensors(values);
  outNative.clear();
  outNative.reserve(tensors.size());
  static auto nativeBaseType = nb::type<NativeDeviceTensor>();
  PyTypeObject *baseType = nativeBaseType.is_valid()
                               ? reinterpret_cast<PyTypeObject *>(nativeBaseType.ptr())
                               : nullptr;
  for (nb::handle tensor : tensors) {
    PyObject *ptr = tensor.ptr();
    if (!nb::isinstance(tensor, tensorType))
      throw CUDAError("program ports require DeviceTensor values");
    if (!baseType || !PyObject_TypeCheck(ptr, baseType))
      return false;
    auto *nd = nb::inst_ptr<NativeDeviceTensor>(ptr);
    if (!nd->nativeBuffer)
      return false;
    outNative.push_back(nd);
  }
  outTensors = std::move(tensors);
  return true;
}

static inline void fastValidateAndPrepare(const std::vector<NativeDeviceTensor *> &nativeTensors,
                                          nb::tuple tensors,
                                          nb::handle session,
                                          CUDASessionQueue *rt) {
  prepareAndValidateExtracted(nativeTensors, std::move(tensors), session, rt);
}

namespace {

std::vector<nb::object> runFastCore(
    PreparedProgram &program,
    const std::vector<TensorPort> &outputs,
    CUDASessionQueue *rt,
    nb::handle session,
    nb::tuple tensors,
    const std::vector<NativeDeviceTensor *> &nativeTensors,
    nb::handle tensorType) {
  if (rt->session->closed())
    throw CUDAError("CUDA execution session is closed");
  fastValidateAndPrepare(nativeTensors, tensors, session, rt);
  rt->reapIfFull(64);
  std::vector<std::shared_ptr<DeviceBuffer>> bound;
  bound.reserve(nativeTensors.size());
  for (auto *nd : nativeTensors) bound.push_back(nd->nativeBuffer);

  std::pair<std::vector<std::shared_ptr<DeviceBuffer>>, std::shared_ptr<Event>> result;
  {
    nb::gil_scoped_release release;
    result = program.run(bound);
  }
  rt->retainAsk(std::move(result.second), tensors);

  std::vector<nb::object> wrapped;
  wrapped.reserve(result.first.size());
  for (std::size_t index = 0; index < result.first.size(); ++index) {
    const auto &buffer = result.first[index];
    nb::object handle;
    for (std::size_t b = 0; b < bound.size(); ++b) {
      if (bound[b].get() == buffer.get()) {
        handle = nb::borrow<nb::object>(tensors[b]);
        break;
      }
    }
    if (!handle.is_valid()) {
      for (std::size_t j = 0; j < index; ++j) {
        if (result.first[j].get() == buffer.get()) {
          handle = nb::borrow<nb::object>(wrapped[j]);
          break;
        }
      }
    }
    if (!handle.is_valid()) {
      const auto &port = outputs[index];
      handle = NativeDeviceTensor::makeSubclass(
          tensorType, rt->sessionWeakRef, buffer, port.shape, port.dtype,
          port.shapeVec, port.dtypeStr, rt->sessionPtr, rt->descriptorStream);
    }
    wrapped.push_back(std::move(handle));
  }
  return wrapped;
}

}  // namespace

std::vector<nb::object> ProgramPorts::runProjected(
    PreparedProgram &program, nb::handle session,
    CUDASessionQueue *rt, const std::vector<nb::object> &allLeaves,
    const std::vector<std::size_t> &indices, bool identityProjection,
    nb::handle tensorType) const {
  std::size_t count = identityProjection ? allLeaves.size() : indices.size();
  if (count != inputs.size())
    throw CUDAError("program expects " + std::to_string(inputs.size()) + " inputs");

  static auto nativeBaseType = nb::type<NativeDeviceTensor>();
  PyTypeObject *baseType = nativeBaseType.is_valid()
                               ? reinterpret_cast<PyTypeObject *>(nativeBaseType.ptr())
                               : nullptr;
  std::vector<NativeDeviceTensor *> nativeTensors;
  nativeTensors.reserve(count);
  nb::tuple tensors = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(count)));
  if (!tensors.ptr()) throw nb::python_error();
  bool allFast = (rt != nullptr && rt->session != nullptr && baseType != nullptr);
  for (std::size_t i = 0; i < count; ++i) {
    const nb::object &obj = allLeaves[identityProjection ? i : indices[i]];
    Py_INCREF(obj.ptr());
    PyTuple_SET_ITEM(tensors.ptr(), static_cast<Py_ssize_t>(i), obj.ptr());
    if (allFast) {
      if (!nb::isinstance(obj, tensorType))
        throw CUDAError("program ports require DeviceTensor values");
      if (!PyObject_TypeCheck(obj.ptr(), baseType)) {
        allFast = false;
      } else {
        auto *nd = nb::inst_ptr<NativeDeviceTensor>(obj.ptr());
        if (!nd->nativeBuffer) {
          allFast = false;
        } else {
          if (nd->shapeVec != inputs[i].shapeVec || nd->dtypeStr != inputs[i].dtypeStr)
            throw CUDAError("program port shape/dtype differs from its compiled type");
          nativeTensors.push_back(nd);
        }
      }
    }
  }
  if (allFast) {
    return runFastCore(program, outputs, rt, session, std::move(tensors), nativeTensors, tensorType);
  }
  nb::list fallback = run(program, session, tensors, tensorType);
  std::vector<nb::object> out;
  out.reserve(nb::len(fallback));
  for (nb::handle item : fallback) out.push_back(nb::borrow<nb::object>(item));
  return out;
}

nb::list ProgramPorts::run(PreparedProgram &program, nb::handle session,
                           nb::handle values, nb::handle tensorType) const {
  CUDASessionQueue *rt = extractSessionQueue(session);
  nb::tuple tensors;
  std::vector<NativeDeviceTensor *> nativeTensors;
  bool fast = (rt != nullptr && rt->session != nullptr &&
               tryFastCheckPorts(values, inputs, "inputs", tensorType, tensors, nativeTensors));

  if (fast) {
    auto vec = runFastCore(program, outputs, rt, session, std::move(tensors), nativeTensors, tensorType);
    nb::list wrapped;
    for (auto &item : vec) wrapped.append(std::move(item));
    return wrapped;
  }

  tensors = slowCheckPorts(values, inputs, "inputs", tensorType);
  slowPrepareAccess(tensors, session);
  auto pending = session.attr("_pending");
  if (PyList_GET_SIZE(pending.ptr()) >= 64) session.attr("reap")();
  auto bound = slowBuffers(tensors);
  std::pair<std::vector<std::shared_ptr<DeviceBuffer>>, std::shared_ptr<Event>> result;
  {
    nb::gil_scoped_release release;
    result = program.run(bound);
  }
  session.attr("_retain")(nb::cast(result.second), tensors);
  nb::list wrapped;
  for (std::size_t index = 0; index < result.first.size(); ++index) {
    const auto &buffer = result.first[index];
    nb::object handle;
    for (std::size_t b = 0; b < bound.size(); ++b) {
      if (bound[b].get() == buffer.get()) {
        handle = nb::borrow<nb::object>(tensors[b]);
        break;
      }
    }
    if (!handle.is_valid()) {
      for (std::size_t j = 0; j < index; ++j) {
        if (result.first[j].get() == buffer.get()) {
          handle = nb::borrow<nb::object>(wrapped[j]);
          break;
        }
      }
    }
    if (!handle.is_valid()) {
      const auto &port = outputs[index];
      handle = tensorType(session, nb::cast(buffer), port.shape, port.dtype);
    }
    wrapped.append(handle);
  }
  return wrapped;
}

nb::object ProgramPorts::submit(PreparedProgram &program, nb::handle session,
                                nb::handle inputValues, nb::handle outputValues,
                                nb::handle tensorType) const {
  CUDASessionQueue *rt = extractSessionQueue(session);
  nb::tuple inputTensors, outputTensors;
  std::vector<NativeDeviceTensor *> nativeInputs, nativeOutputs;
  bool fastIn = (rt != nullptr && rt->session != nullptr &&
                 tryFastCheckPorts(inputValues, inputs, "inputs", tensorType, inputTensors, nativeInputs));
  bool fastOut = (fastIn &&
                  tryFastCheckPorts(outputValues, outputs, "outputs", tensorType, outputTensors, nativeOutputs));

  if (fastOut) {
    if (rt->session->closed())
      throw CUDAError("CUDA execution session is closed");
    fastValidateAndPrepare(nativeInputs, inputTensors, session, rt);
    fastValidateAndPrepare(nativeOutputs, outputTensors, session, rt);
    rt->reapIfFull(64);
    std::vector<std::shared_ptr<DeviceBuffer>> inputBuffers, outputBuffers;
    inputBuffers.reserve(nativeInputs.size());
    outputBuffers.reserve(nativeOutputs.size());
    for (auto *nd : nativeInputs) inputBuffers.push_back(nd->nativeBuffer);
    for (auto *nd : nativeOutputs) outputBuffers.push_back(nd->nativeBuffer);
    std::shared_ptr<Event> event;
    {
      nb::gil_scoped_release release;
      event = program.submit(inputBuffers, outputBuffers);
    }
    auto references = nb::steal<nb::object>(PySequence_Concat(inputTensors.ptr(), outputTensors.ptr()));
    if (!references.is_valid()) throw nb::python_error();
    return session.attr("_retain")(nb::cast(event), references);
  }

  inputTensors = slowCheckPorts(inputValues, inputs, "inputs", tensorType);
  outputTensors = slowCheckPorts(outputValues, outputs, "outputs", tensorType);
  slowPrepareAccess(inputTensors, session);
  slowPrepareAccess(outputTensors, session);
  auto pending = session.attr("_pending");
  if (PyList_GET_SIZE(pending.ptr()) >= 64) session.attr("reap")();
  auto inputBuffers = slowBuffers(inputTensors), outputBuffers = slowBuffers(outputTensors);
  std::shared_ptr<Event> event;
  {
    nb::gil_scoped_release release;
    event = program.submit(inputBuffers, outputBuffers);
  }
  auto references = nb::steal<nb::object>(PySequence_Concat(inputTensors.ptr(), outputTensors.ptr()));
  if (!references.is_valid()) throw nb::python_error();
  return session.attr("_retain")(nb::cast(event), references);
}

}  // namespace fuga::ask_rt

void bindAskRuntime(nanobind::module_ &module) {
  using namespace ask::runtime;
  using namespace fuga::ask_rt;

  nb::exception<CUDAError>(
      module, "CUDAError",
      nb::module_::import_("fugatensor._ask.cuda").attr("CUDAError"));

  static const PyType_Slot dtSlots[] = {
      {Py_tp_traverse, reinterpret_cast<void *>(deviceTensorTraverse)},
      {Py_tp_clear, reinterpret_cast<void *>(deviceTensorClear)},
      {0, nullptr},
  };
  nb::class_<NativeDeviceTensor>(module, "NativeDeviceTensor", nb::type_slots(dtSlots))
      .def("__init__", [](NativeDeviceTensor *self,
                          nb::object buffer,
                          nb::tuple shapeTuple,
                          nb::str dtypePyStr,
                          bool hasOwner,
                          std::uintptr_t sessionPtr,
                          nb::object sessionRef,
                          nb::object owner,
                          nb::object snapshot,
                          nb::object producerStream) {
        new (self) NativeDeviceTensor();
        if (!buffer.is_none() && nb::isinstance<DeviceBuffer>(buffer)) {
          self->nativeBuffer = nb::cast<std::shared_ptr<DeviceBuffer>>(buffer);
        } else {
          self->pyBuffer = std::move(buffer);
        }
        self->shapeTuple = std::move(shapeTuple);
        self->dtypePyStr = std::move(dtypePyStr);
        self->shapeVec = nb::cast<std::vector<std::int64_t>>(self->shapeTuple);
        self->dtypeStr = nb::cast<std::string>(self->dtypePyStr);
        self->sessionPtr = sessionPtr;
        self->hasOwner = hasOwner;
        self->sessionRef = std::move(sessionRef);
        self->owner = std::move(owner);
        self->snapshot = std::move(snapshot);
        self->producerStream = std::move(producerStream);
      }, nb::arg("buffer").none(), nb::arg("shape"), nb::arg("dtype"),
         nb::arg("has_owner"), nb::arg("session_ptr"),
         nb::arg("session_ref").none(), nb::arg("owner").none(),
         nb::arg("snapshot").none(), nb::arg("producer_stream").none())
      .def_prop_ro("_shape", [](const NativeDeviceTensor &self) { return self.shapeTuple; })
      .def_prop_ro("_dtype", [](const NativeDeviceTensor &self) { return self.dtypePyStr; })
      .def_prop_ro("shape", [](const NativeDeviceTensor &self) { return self.shapeTuple; })
      .def_prop_ro("dtype", [](const NativeDeviceTensor &self) { return self.dtypePyStr; })
      .def_prop_ro("_buffer", [](const NativeDeviceTensor &self) -> nb::object {
        return self.nativeBuffer ? nb::cast(self.nativeBuffer) : self.pyBuffer;
      })
      .def_prop_ro("_session_ref", [](const NativeDeviceTensor &self) { return self.sessionRef; })
      .def_rw("_owner", &NativeDeviceTensor::owner)
      .def_rw("_snapshot", &NativeDeviceTensor::snapshot)
      .def_rw("_producer_stream", &NativeDeviceTensor::producerStream)
      .def_prop_ro("nbytes", [](const NativeDeviceTensor &self) -> nb::object {
        if (self.nativeBuffer) return nb::int_(self.nativeBuffer->size());
        return self.pyBuffer.attr("nbytes");
      })
      .def_prop_ro("data_ptr", [](const NativeDeviceTensor &self) -> nb::object {
        if (self.nativeBuffer) return nb::int_(self.nativeBuffer->pointer());
        return self.pyBuffer.attr("data_ptr");
      })
      .def_prop_ro("readonly", [](const NativeDeviceTensor &self) -> nb::object {
        if (self.nativeBuffer) return nb::bool_(self.nativeBuffer->readonly());
        return self.pyBuffer.attr("readonly");
      });

  static const PyType_Slot sqSlots[] = {
      {Py_tp_traverse, reinterpret_cast<void *>(sessionQueueTraverse)},
      {Py_tp_clear, reinterpret_cast<void *>(sessionQueueClear)},
      {0, nullptr},
  };
  nb::class_<CUDASessionQueue>(module, "CUDASessionQueue", nb::type_slots(sqSlots))
      .def(nb::init<std::shared_ptr<Session>, nb::object, nb::object>())
      .def_ro("session_ptr", &CUDASessionQueue::sessionPtr)
      .def("reap", &CUDASessionQueue::reap)
      .def("clear_pending", &CUDASessionQueue::clearPending)
      .def("make_tensor", [](CUDASessionQueue &self,
                             nb::handle tensorType,
                             std::shared_ptr<DeviceBuffer> buffer,
                             std::vector<std::int64_t> shapeVec,
                             std::string dtypeStr) {
        nb::tuple shapeTuple(nb::cast(shapeVec));
        nb::str dtypePyStr(dtypeStr.c_str());
        return NativeDeviceTensor::makeSubclass(
            tensorType, self.sessionWeakRef, std::move(buffer),
            std::move(shapeTuple), std::move(dtypePyStr),
            std::move(shapeVec), std::move(dtypeStr),
            self.sessionPtr, self.descriptorStream);
      });

  nb::class_<DeviceBuffer>(module, "CUDADeviceBuffer")
      .def_prop_ro("data_ptr", &DeviceBuffer::pointer)
      .def_prop_ro("nbytes", &DeviceBuffer::size)
      .def_prop_ro("readonly", &DeviceBuffer::readonly)
      .def("copy_from_host", [](DeviceBuffer &buffer, nb::handle data) {
        BufferView source(data, false);
        nb::gil_scoped_release release;
        buffer.copyFromHost(source.view.buf, static_cast<std::size_t>(source.view.len));
      })
      .def("copy_to_host", [](const DeviceBuffer &buffer, nb::handle destination) {
        BufferView target(destination, true);
        nb::gil_scoped_release release;
        buffer.copyToHost(target.view.buf, static_cast<std::size_t>(target.view.len));
      })
      .def("to_host_bytes", [](const DeviceBuffer &buffer) {
        std::string data;
        {
          nb::gil_scoped_release release;
          data = buffer.copyToHost();
        }
        return nb::bytes(data.data(), data.size());
      });

  nb::class_<Kernel>(module, "CUDAKernel")
      .def_prop_ro("entry", &Kernel::entry)
      .def_prop_ro("max_threads", &Kernel::maxThreads);

  nb::class_<Event>(module, "CUDAEvent")
      .def("query", &Event::query, nb::call_guard<nb::gil_scoped_release>())
      .def("wait", &Event::wait, nb::call_guard<nb::gil_scoped_release>());

  nb::class_<PreparedProgram>(module, "CUDAProgram")
      .def("submit", &PreparedProgram::submit, nb::arg("inputs"), nb::arg("outputs"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("run", &PreparedProgram::run, nb::arg("inputs"), nb::call_guard<nb::gil_scoped_release>());

  nb::class_<ProgramPorts>(module, "CUDAProgramPorts")
      .def(nb::init<nb::sequence, nb::sequence>(), nb::arg("inputs"), nb::arg("outputs"))
      .def("run", &ProgramPorts::run, nb::arg("program"), nb::arg("session"),
           nb::arg("inputs").none(), nb::arg("tensor_type"))
      .def("submit", &ProgramPorts::submit, nb::arg("program"), nb::arg("session"),
           nb::arg("inputs").none(), nb::arg("outputs").none(), nb::arg("tensor_type"));

  nb::class_<Session>(module, "CUDASession")
      .def_static("create", &Session::create, nb::arg("device") = 0,
                  nb::call_guard<nb::gil_scoped_release>())
      .def("allocate", &Session::allocate, nb::arg("nbytes"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("import_buffer", &Session::importBuffer, nb::arg("pointer"), nb::arg("nbytes"),
           nb::arg("readonly") = false, nb::call_guard<nb::gil_scoped_release>())
      .def("load", [](Session &session, nb::bytes data, const std::string &entry,
                      const std::vector<std::string> &types, int arch,
                      const std::vector<unsigned> &widths) {
        std::string binary(static_cast<const char *>(data.data()), data.size());
        nb::gil_scoped_release release;
        return session.load(binary, entry, types, arch, widths);
      }, nb::arg("cubin"), nb::arg("entry"), nb::arg("required_types"), nb::arg("arch"),
         nb::arg("parameter_widths"))
      .def("submit", [](Session &session, const std::shared_ptr<Kernel> &kernel,
                        nb::list bindings, std::array<unsigned, 3> grid,
                        std::array<unsigned, 3> block, unsigned sharedBytes) {
        std::vector<KernelArgument> arguments;
        for (nb::handle binding : bindings) {
          KernelArgument argument;
          if (nb::isinstance<nb::bytes>(binding)) {
            auto bytes = nb::cast<nb::bytes>(binding);
            argument.scalar.assign(static_cast<const char *>(bytes.data()), bytes.size());
            if (argument.scalar.empty()) throw CUDAError("scalar argument cannot be empty");
          } else if (!binding.is_none()) {
            argument.buffer = nb::cast<std::shared_ptr<DeviceBuffer>>(binding);
          }
          arguments.push_back(std::move(argument));
        }
        nb::gil_scoped_release release;
        return session.submit(kernel, arguments, grid, block, sharedBytes);
      }, nb::arg("kernel"), nb::arg("bindings"), nb::arg("grid"), nb::arg("block"),
         nb::arg("shared_bytes") = 0)
      .def("prepare_program", [](Session &session, std::vector<std::size_t> sizes,
                                 std::vector<std::size_t> inputs, std::vector<std::size_t> outputs,
                                 nb::list definitions, bool graph) {
        std::vector<PlanStep> steps;
        for (nb::handle definition : definitions) {
          auto item = nb::cast<nb::dict>(definition);
          PlanStep step;
          step.inputs = nb::cast<std::vector<std::size_t>>(item["inputs"]);
          step.outputs = nb::cast<std::vector<std::size_t>>(item["outputs"]);
          if (item.contains("matmul")) {
            auto description = nb::cast<nb::dict>(item["matmul"]);
            MatmulDescription matmul;
            matmul.elementType = nb::cast<std::string>(description["element_type"]);
            matmul.m = nb::cast<std::uint64_t>(description["m"]);
            matmul.n = nb::cast<std::uint64_t>(description["n"]);
            matmul.k = nb::cast<std::uint64_t>(description["k"]);
            matmul.batchCount = nb::cast<std::uint64_t>(description["batch_count"]);
            matmul.transposeA = nb::cast<bool>(description["transpose_a"]);
            matmul.transposeB = nb::cast<bool>(description["transpose_b"]);
            if (description.contains("workspace_limit"))
              matmul.workspaceLimit = nb::cast<std::size_t>(description["workspace_limit"]);
            step.matmul = std::move(matmul);
            if (item.contains("pre_copy")) {
            auto pair = nb::cast<std::pair<std::size_t, std::size_t>>(item["pre_copy"]);
            step.preCopy = CopyDescription{pair.first, pair.second};
          }
          steps.push_back(std::move(step));
            continue;
          }
          step.kernel = nb::cast<std::shared_ptr<Kernel>>(item["kernel"]);
          step.grid = nb::cast<std::array<unsigned, 3>>(item["grid"]);
          step.block = nb::cast<std::array<unsigned, 3>>(item["block"]);
          step.sharedBytes = nb::cast<unsigned>(item["shared_bytes"]);
          for (nb::handle binding : nb::cast<nb::list>(item["arguments"])) {
            PlanArgument argument;
            if (nb::isinstance<nb::bytes>(binding)) {
              auto scalar = nb::cast<nb::bytes>(binding);
              argument.scalar.assign(static_cast<const char *>(scalar.data()), scalar.size());
              if (argument.scalar.empty()) throw CUDAError("scalar argument cannot be empty");
            } else if (!binding.is_none()) {
              argument.buffer = nb::cast<std::int64_t>(binding);
            }
            step.arguments.push_back(std::move(argument));
          }
          if (item.contains("pre_copy")) {
            auto pair = nb::cast<std::pair<std::size_t, std::size_t>>(item["pre_copy"]);
            step.preCopy = CopyDescription{pair.first, pair.second};
          }
          steps.push_back(std::move(step));
        }
        nb::gil_scoped_release release;
        return session.prepareProgram(std::move(sizes), std::move(inputs), std::move(outputs),
                                      std::move(steps), graph);
      }, nb::arg("buffer_sizes"), nb::arg("inputs"), nb::arg("outputs"), nb::arg("steps"),
         nb::arg("graph") = false)
      .def("wait_for_external_stream", &Session::waitForExternalStream,
           nb::call_guard<nb::gil_scoped_release>())
      .def("check_types", &Session::checkTypes, nb::call_guard<nb::gil_scoped_release>())
      .def("synchronize", &Session::synchronize, nb::call_guard<nb::gil_scoped_release>())
      .def("reap", &Session::reap, nb::call_guard<nb::gil_scoped_release>())
      .def("close", &Session::close, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("closed", &Session::closed, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("stream", &Session::stream, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("device", &Session::device)
      .def_prop_ro("capability", &Session::capability)
      .def_prop_ro("device_name", &Session::deviceName)
      .def_prop_ro("stats", &Session::stats, nb::call_guard<nb::gil_scoped_release>());
}
