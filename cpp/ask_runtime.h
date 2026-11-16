#pragma once

#include "ask/Runtime/CUDA.h"
#include "cuda_driver_api.h"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fuga::ask_rt {
namespace nb = nanobind;

struct TensorPort {
  nb::tuple shape;
  nb::str dtype;
  std::vector<std::int64_t> shapeVec;
  std::string dtypeStr;
};

struct NativeDeviceTensor {
  std::shared_ptr<ask::runtime::DeviceBuffer> nativeBuffer;
  std::vector<std::int64_t> shapeVec;
  std::string dtypeStr;
  std::uintptr_t sessionPtr = 0;
  bool hasOwner = false;
  nb::tuple shapeTuple;
  nb::str dtypePyStr;
  nb::object pyBuffer;
  nb::object sessionRef;
  nb::object owner;
  nb::object snapshot;
  nb::object producerStream;

  static nb::object makeSubclass(nb::handle tensorType,
                                 nb::handle sessionRef,
                                 std::shared_ptr<ask::runtime::DeviceBuffer> buffer,
                                 nb::tuple shapeTuple,
                                 nb::str dtypePyStr,
                                 std::vector<std::int64_t> shapeVec,
                                 std::string dtypeStr,
                                 std::uintptr_t sessionPtr,
                                 nb::handle producerStream);
};

class CUDASessionQueue {
public:
  struct PendingEntry {
    std::shared_ptr<ask::runtime::Event> askEvent;
    cuda_drv::CUevent cuEvent = nullptr;
    nb::object references;
    std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> buffers;
  };

  CUDASessionQueue(std::shared_ptr<ask::runtime::Session> session,
                   nb::object sessionWeakRef,
                   nb::object descriptorStream);
  ~CUDASessionQueue();

  CUDASessionQueue(const CUDASessionQueue &) = delete;
  CUDASessionQueue &operator=(const CUDASessionQueue &) = delete;

  void retainAsk(std::shared_ptr<ask::runtime::Event> event, nb::object references,
                 std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> buffers = {});
  void retainStreamWork(nb::object references,
                        std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> buffers = {});
  void reap();
  void reapIfFull(std::size_t threshold = 64);
  void clearPending();

  std::shared_ptr<ask::runtime::Session> session;
  nb::object sessionWeakRef;
  nb::object descriptorStream;
  std::uintptr_t sessionPtr = 0;
  cuda_drv::CUcontext cudaCtx = nullptr;
  std::vector<PendingEntry> pending;
  std::vector<cuda_drv::CUevent> eventPool;
};

CUDASessionQueue *extractSessionQueue(nb::handle session);
std::size_t elemByteWidth(const std::string &dtype);
bool extractInputsForSession(nb::handle values,
                             nb::handle tensorType,
                             nb::handle session,
                             CUDASessionQueue *rt,
                             nb::tuple &outTensors,
                             std::vector<NativeDeviceTensor *> &outNative);
void prepareAndValidateExtracted(const std::vector<NativeDeviceTensor *> &nativeTensors,
                                 nb::tuple tensors,
                                 nb::handle session,
                                 CUDASessionQueue *rt);

class ProgramPorts {
public:
  ProgramPorts(nb::sequence inputs, nb::sequence outputs);

  nb::list run(ask::runtime::PreparedProgram &program, nb::handle session,
               nb::handle values, nb::handle tensorType) const;

  std::vector<nb::object> runProjected(
      ask::runtime::PreparedProgram &program, nb::handle session,
      CUDASessionQueue *rt, const std::vector<nb::object> &allLeaves,
      const std::vector<std::size_t> &indices, bool identityProjection,
      nb::handle tensorType) const;

  nb::object submit(ask::runtime::PreparedProgram &program, nb::handle session,
                    nb::handle inputValues, nb::handle outputValues,
                    nb::handle tensorType) const;

  // Fast internal execution over pre-extracted DeviceBuffer list without Python wrapping.
  // Throws on port mismatch; caller still validates/retains at boundaries.
  const std::vector<TensorPort> &inputPorts() const { return inputs; }
  const std::vector<TensorPort> &outputPorts() const { return outputs; }
  static std::vector<TensorPort> readPorts(nb::sequence definitions);

private:
  std::vector<TensorPort> inputs, outputs;
};

}  // namespace fuga::ask_rt
