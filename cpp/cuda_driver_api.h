#pragma once

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fuga::cuda_drv {

using CUresult = int;
using CUstream = void *;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUevent = void *;
using CUdeviceptr = std::uint64_t;

inline constexpr CUresult kCudaSuccess = 0;
inline constexpr CUresult kCudaErrorNotReady = 600;
inline constexpr unsigned kCuEventDisableTiming = 0x02;

struct DriverApi {
  using fnCuCtxGetCurrent = CUresult (*)(CUcontext *);
  using fnCuCtxPushCurrent = CUresult (*)(CUcontext);
  using fnCuCtxPopCurrent = CUresult (*)(CUcontext *);
  using fnCuStreamGetCtx = CUresult (*)(CUstream, CUcontext *);
  using fnCuStreamSynchronize = CUresult (*)(CUstream);
  using fnCuEventCreate = CUresult (*)(CUevent *, unsigned);
  using fnCuEventDestroy = CUresult (*)(CUevent);
  using fnCuEventRecord = CUresult (*)(CUevent, CUstream);
  using fnCuEventQuery = CUresult (*)(CUevent);
  using fnCuEventSynchronize = CUresult (*)(CUevent);
  using fnCuModuleLoadData = CUresult (*)(CUmodule *, const void *);
  using fnCuModuleUnload = CUresult (*)(CUmodule);
  using fnCuModuleGetFunction = CUresult (*)(CUfunction *, CUmodule, const char *);
  using fnCuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned,
                                        CUstream, void **, void **);
  using fnCuMemcpyDtoHAsync = CUresult (*)(void *, CUdeviceptr, std::size_t, CUstream);
  using fnCuMemsetD8Async = CUresult (*)(CUdeviceptr, unsigned char, std::size_t, CUstream);

  void *lib = nullptr;
  fnCuCtxGetCurrent cuCtxGetCurrent = nullptr;
  fnCuCtxPushCurrent cuCtxPushCurrent = nullptr;
  fnCuCtxPopCurrent cuCtxPopCurrent = nullptr;
  fnCuStreamGetCtx cuStreamGetCtx = nullptr;
  fnCuStreamSynchronize cuStreamSynchronize = nullptr;
  fnCuEventCreate cuEventCreate = nullptr;
  fnCuEventDestroy cuEventDestroy = nullptr;
  fnCuEventRecord cuEventRecord = nullptr;
  fnCuEventQuery cuEventQuery = nullptr;
  fnCuEventSynchronize cuEventSynchronize = nullptr;
  fnCuModuleLoadData cuModuleLoadData = nullptr;
  fnCuModuleUnload cuModuleUnload = nullptr;
  fnCuModuleGetFunction cuModuleGetFunction = nullptr;
  fnCuLaunchKernel cuLaunchKernel = nullptr;
  fnCuMemcpyDtoHAsync cuMemcpyDtoHAsync = nullptr;
  fnCuMemsetD8Async cuMemsetD8Async = nullptr;
};

inline const DriverApi &driver() {
  static DriverApi api;
  static std::once_flag once;
  static std::string err;
  std::call_once(once, [&]() {
    api.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!api.lib)
      api.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!api.lib) {
      const char *dlerr = dlerror();
      err = std::string("failed to load libcuda.so.1: ") + (dlerr ? dlerr : "unknown error");
      return;
    }
    auto loadSym = [&](auto &dst, const char *primary, const char *fallback = nullptr) {
      void *sym = dlsym(api.lib, primary);
      if (!sym && fallback)
        sym = dlsym(api.lib, fallback);
      if (!sym) {
        err = std::string("missing CUDA driver symbol: ") + primary;
        return false;
      }
      dst = reinterpret_cast<std::decay_t<decltype(dst)>>(sym);
      return true;
    };
    loadSym(api.cuCtxGetCurrent, "cuCtxGetCurrent") &&
        loadSym(api.cuCtxPushCurrent, "cuCtxPushCurrent_v2", "cuCtxPushCurrent") &&
        loadSym(api.cuCtxPopCurrent, "cuCtxPopCurrent_v2", "cuCtxPopCurrent") &&
        loadSym(api.cuStreamGetCtx, "cuStreamGetCtx") &&
        loadSym(api.cuStreamSynchronize, "cuStreamSynchronize") &&
        loadSym(api.cuEventCreate, "cuEventCreate") &&
        loadSym(api.cuEventDestroy, "cuEventDestroy_v2", "cuEventDestroy") &&
        loadSym(api.cuEventRecord, "cuEventRecord") &&
        loadSym(api.cuEventQuery, "cuEventQuery") &&
        loadSym(api.cuEventSynchronize, "cuEventSynchronize") &&
        loadSym(api.cuModuleLoadData, "cuModuleLoadData") &&
        loadSym(api.cuModuleUnload, "cuModuleUnload") &&
        loadSym(api.cuModuleGetFunction, "cuModuleGetFunction") &&
        loadSym(api.cuLaunchKernel, "cuLaunchKernel") &&
        loadSym(api.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync") &&
        loadSym(api.cuMemsetD8Async, "cuMemsetD8Async");
  });
  if (!err.empty())
    throw std::runtime_error(err);
  return api;
}

inline CUcontext streamContext(std::uint64_t rawStream) {
  const auto &drv = driver();
  CUcontext ctx = nullptr;
  CUresult rc = drv.cuStreamGetCtx(reinterpret_cast<CUstream>(rawStream), &ctx);
  if (rc != kCudaSuccess || !ctx)
    throw std::runtime_error("cuStreamGetCtx failed with error " + std::to_string(rc));
  return ctx;
}

class ScopedCudaContext {
public:
  explicit ScopedCudaContext(CUcontext target) : drv_(driver()) {
    if (!target)
      return;
    CUcontext cur = nullptr;
    if (drv_.cuCtxGetCurrent(&cur) == kCudaSuccess && cur == target)
      return;
    CUresult rc = drv_.cuCtxPushCurrent(target);
    if (rc != kCudaSuccess)
      throw std::runtime_error("cuCtxPushCurrent failed with error " + std::to_string(rc));
    pushed_ = true;
  }
  ~ScopedCudaContext() {
    if (pushed_) {
      CUcontext popped = nullptr;
      drv_.cuCtxPopCurrent(&popped);
    }
  }
  ScopedCudaContext(const ScopedCudaContext &) = delete;
  ScopedCudaContext &operator=(const ScopedCudaContext &) = delete;

private:
  const DriverApi &drv_;
  bool pushed_ = false;
};

}  // namespace fuga::cuda_drv
