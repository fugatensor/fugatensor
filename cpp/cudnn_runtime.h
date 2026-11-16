#pragma once

#include "ask_runtime.h"
#include "cuda_driver_api.h"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fuga::cudnn_rt {
namespace nb = nanobind;

bool cudnnAvailable();

struct CudnnPlan;

struct CudnnPlanKey {
  std::string entry;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<std::string> dtypes;
  bool causal = false;
  std::uint64_t scaleBits = 0;
  std::int64_t maxSq = 0;
  std::int64_t maxSkv = 0;
  std::string layout;

  bool operator==(const CudnnPlanKey &other) const;
};

struct CudnnPlanKeyHash {
  std::size_t operator()(const CudnnPlanKey &k) const;
};

struct BwdAuxFunctions {
  cuda_drv::CUmodule module = nullptr;
  cuda_drv::CUfunction rowsumF16 = nullptr;
  cuda_drv::CUfunction rowsumBf16 = nullptr;
  cuda_drv::CUfunction softmaxScaleF16 = nullptr;
  cuda_drv::CUfunction softmaxScaleBf16 = nullptr;
  cuda_drv::CUfunction varlenMeta = nullptr;
  cuda_drv::CUfunction varlenUnpack16 = nullptr;
  cuda_drv::CUfunction varlenPack16 = nullptr;
};

class CudnnSessionState {
public:
  explicit CudnnSessionState(std::shared_ptr<ask_rt::CUDASessionQueue> queue);
  ~CudnnSessionState();

  CudnnSessionState(const CudnnSessionState &) = delete;
  CudnnSessionState &operator=(const CudnnSessionState &) = delete;

  void clearCache();
  void close();
  std::size_t cacheSize() const { return plans.size(); }

  std::shared_ptr<ask::runtime::DeviceBuffer> scratch(std::size_t slot, std::size_t nbytes);
  const BwdAuxFunctions &auxFunctions();
  std::shared_ptr<CudnnPlan> getOrCreatePlan(
      const std::string &entry,
      const std::vector<std::vector<std::int64_t>> &shapes,
      const std::vector<std::string> &dtypes,
      bool causal,
      bool hasScale,
      double scale,
      std::int64_t maxSq,
      std::int64_t maxSkv,
      const std::string &layout);

  std::shared_ptr<ask_rt::CUDASessionQueue> queue;
  std::shared_ptr<ask::runtime::Session> session;
  void *cudnnHandle = nullptr;
  int arch = 80;
  bool op43Unsupported = false;
  bool closed = false;
  BwdAuxFunctions aux{};
  std::unordered_map<CudnnPlanKey, std::shared_ptr<CudnnPlan>, CudnnPlanKeyHash> plans;
  std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> scratchPool;
};

class CudnnExecutable {
public:
  struct OutputSpec {
    std::vector<std::int64_t> shape;
    std::string dtype;
    nb::tuple shapeTuple;
    nb::str dtypePyStr;
  };

  CudnnExecutable(std::shared_ptr<CudnnSessionState> state,
                  nb::object sessionObj,
                  nb::object tensorType,
                  std::string entry,
                  std::vector<OutputSpec> outputSpecs,
                  bool causal,
                  bool hasScale,
                  double scale,
                  std::int64_t maxSq,
                  std::int64_t maxSkv,
                  std::string layout,
                  nb::object artifactDir,
                  std::string opId);

  nb::list run(nb::handle inputsObj);

  // Raw buffer execution cleanly callable without holding the GIL.
  void executeRaw(const std::vector<ask::runtime::DeviceBuffer *> &inBufs,
                  const std::vector<std::vector<std::int64_t>> &inShapes,
                  const std::vector<std::string> &inDtypes,
                  std::vector<std::shared_ptr<ask::runtime::DeviceBuffer>> &outBufs,
                  std::vector<std::vector<std::int64_t>> &outShapes,
                  std::vector<std::string> &outDtypes);

  std::shared_ptr<CudnnSessionState> state;
  nb::object sessionObj;
  nb::object tensorType;
  std::string entry;
  std::vector<OutputSpec> outputSpecs;
  bool causal = false;
  bool hasScale = false;
  double scale = 0.0;
  std::int64_t maxSq = 0;
  std::int64_t maxSkv = 0;
  std::string layout;
  std::string opId;
  std::string sourceStr;
  nb::str sourcePy;
  nb::object artifactDir;
  nb::dict reportDict;
  nb::dict executionPlanDict;
};

class CudnnCompositeExecutable {
public:
  struct Stage {
    bool isExternal = false;
    std::string id;
    std::vector<std::size_t> inputSlots;
    std::vector<std::size_t> outputSlots;
    std::shared_ptr<ask::runtime::PreparedProgram> program;
    std::shared_ptr<CudnnExecutable> cudnn;
    std::vector<ask_rt::TensorPort> externalOutputs;
  };

  CudnnCompositeExecutable(std::shared_ptr<CudnnSessionState> state,
                           nb::object sessionObj,
                           nb::object tensorType,
                           std::size_t numSlots,
                           std::vector<std::size_t> initialInputSlots,
                           std::vector<std::size_t> finalOutputSlots,
                           std::vector<ask_rt::TensorPort> initialInputPorts,
                           std::vector<ask_rt::TensorPort> finalOutputPorts,
                           std::vector<std::vector<std::int64_t>> slotShapes,
                           std::vector<std::string> slotDtypes,
                           std::vector<Stage> stages,
                           nb::object artifactDir,
                           nb::dict executionPlan,
                           nb::dict report);

  nb::list run(nb::handle inputsObj);

  std::shared_ptr<CudnnSessionState> state;
  nb::object sessionObj;
  nb::object tensorType;
  std::size_t numSlots = 0;
  std::vector<std::size_t> initialInputSlots;
  std::vector<std::size_t> finalOutputSlots;
  std::vector<ask_rt::TensorPort> initialInputPorts;
  std::vector<ask_rt::TensorPort> finalOutputPorts;
  std::vector<std::vector<std::int64_t>> slotShapes;
  std::vector<std::string> slotDtypes;
  std::vector<Stage> stages;
  nb::object artifactDir;
  nb::dict executionPlanDict;
  nb::dict reportDict;
  nb::str sourcePy;
};

}  // namespace fuga::cudnn_rt
