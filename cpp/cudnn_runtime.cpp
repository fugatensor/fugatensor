#include "cudnn_runtime.h"
#include "cudnn_bwd_ptx.h"

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace fuga::cudnn_rt {
using namespace ask::runtime;
using namespace fuga::ask_rt;

namespace {

// Minimal cuDNN 9 C Backend ABI definitions (no build-time cudnn headers required).
using cudnnHandle_t = void *;
using cudnnBackendDescriptor_t = void *;
using cudnnStatus_t = int;

constexpr cudnnStatus_t CUDNN_STATUS_SUCCESS = 0;
constexpr cudnnStatus_t CUDNN_STATUS_NOT_SUPPORTED = 3000;

constexpr int CUDNN_DATA_FLOAT = 0;
constexpr int CUDNN_DATA_HALF = 2;
constexpr int CUDNN_DATA_INT32 = 4;
constexpr int CUDNN_DATA_BFLOAT16 = 9;
constexpr int CUDNN_DATA_INT64 = 10;
constexpr int CUDNN_DATA_BOOLEAN = 11;

constexpr int CUDNN_POINTWISE_CMP_GE = 303;
constexpr int CUDNN_POINTWISE_GEN_INDEX = 501;
constexpr int CUDNN_POINTWISE_BINARY_SELECT = 601;

constexpr int CUDNN_HEUR_MODE_INSTANT = 0;
constexpr int CUDNN_HEUR_MODE_B = 1;
constexpr int CUDNN_HEUR_MODE_FALLBACK = 2;
constexpr int CUDNN_HEUR_MODE_A = 3;

constexpr int CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0;
constexpr int CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 3;
constexpr int CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 4;
constexpr int CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 5;
constexpr int CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 13;
constexpr int CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 15;
constexpr int CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 16;
constexpr int CUDNN_BACKEND_TENSOR_DESCRIPTOR = 17;
constexpr int CUDNN_BACKEND_MATMUL_DESCRIPTOR = 18;
constexpr int CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 19;
constexpr int CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = 41;
constexpr int CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR = 43;

constexpr int CUDNN_TYPE_HANDLE = 0;
constexpr int CUDNN_TYPE_DATA_TYPE = 1;
constexpr int CUDNN_TYPE_BOOLEAN = 2;
constexpr int CUDNN_TYPE_INT64 = 3;
constexpr int CUDNN_TYPE_VOID_PTR = 6;
constexpr int CUDNN_TYPE_HEUR_MODE = 8;
constexpr int CUDNN_TYPE_POINTWISE_MODE = 14;
constexpr int CUDNN_TYPE_BACKEND_DESCRIPTOR = 15;

constexpr int CUDNN_ATTR_POINTWISE_MODE = 0;
constexpr int CUDNN_ATTR_POINTWISE_MATH_PREC = 1;
constexpr int CUDNN_ATTR_POINTWISE_AXIS = 9;

constexpr int CUDNN_ATTR_ENGINEHEUR_MODE = 200;
constexpr int CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201;
constexpr int CUDNN_ATTR_ENGINEHEUR_RESULTS = 202;

constexpr int CUDNN_ATTR_EXECUTION_PLAN_HANDLE = 400;
constexpr int CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 401;
constexpr int CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 402;

constexpr int CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_XDESC = 751;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_BDESC = 752;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_YDESC = 753;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_TDESC = 758;

constexpr int CUDNN_ATTR_OPERATIONGRAPH_HANDLE = 800;
constexpr int CUDNN_ATTR_OPERATIONGRAPH_OPS = 801;

constexpr int CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = 900;
constexpr int CUDNN_ATTR_TENSOR_DATA_TYPE = 901;
constexpr int CUDNN_ATTR_TENSOR_DIMENSIONS = 902;
constexpr int CUDNN_ATTR_TENSOR_STRIDES = 903;
constexpr int CUDNN_ATTR_TENSOR_UNIQUE_ID = 906;
constexpr int CUDNN_ATTR_TENSOR_IS_VIRTUAL = 907;
constexpr int CUDNN_ATTR_TENSOR_IS_BY_VALUE = 908;
constexpr int CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = 913;

constexpr int CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 1000;
constexpr int CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001;
constexpr int CUDNN_ATTR_VARIANT_PACK_WORKSPACE = 1003;

constexpr int CUDNN_ATTR_MATMUL_COMP_TYPE = 1500;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_ADESC = 1520;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_BDESC = 1521;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_CDESC = 1522;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_DESC = 1523;

constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC = 2800;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC = 2801;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC = 2802;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC = 2803;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = 2804;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = 2805;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = 2809;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = 2810;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH = 2811;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID = 2812;
constexpr int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID = 2813;

constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC = 2851;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC = 2852;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC = 2853;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC = 2854;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC = 2855;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC = 2856;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC = 2857;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC = 2858;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC = 2859;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC = 2860;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC = 2861;
constexpr int CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC = 2862;

struct CudnnApi {
  using fnCreate = cudnnStatus_t (*)(cudnnHandle_t *);
  using fnDestroy = cudnnStatus_t (*)(cudnnHandle_t);
  using fnSetStream = cudnnStatus_t (*)(cudnnHandle_t, void *);
  using fnGetErrorString = const char *(*)(cudnnStatus_t);
  using fnGetLastErrorString = void (*)(char *, std::size_t);
  using fnBCreate = cudnnStatus_t (*)(int, cudnnBackendDescriptor_t *);
  using fnBDestroy = cudnnStatus_t (*)(cudnnBackendDescriptor_t);
  using fnBFinalize = cudnnStatus_t (*)(cudnnBackendDescriptor_t);
  using fnBSet = cudnnStatus_t (*)(cudnnBackendDescriptor_t, int, int, std::int64_t, const void *);
  using fnBGet = cudnnStatus_t (*)(cudnnBackendDescriptor_t, int, int, std::int64_t, std::int64_t *, void *);
  using fnBExecute = cudnnStatus_t (*)(cudnnHandle_t, cudnnBackendDescriptor_t, cudnnBackendDescriptor_t);

  fnCreate create = nullptr;
  fnDestroy destroy = nullptr;
  fnSetStream setStream = nullptr;
  fnGetErrorString getErrorString = nullptr;
  fnGetLastErrorString getLastErrorString = nullptr;
  fnBCreate createDesc = nullptr;
  fnBDestroy destroyDesc = nullptr;
  fnBFinalize finalizeDesc = nullptr;
  fnBSet setAttr = nullptr;
  fnBGet getAttr = nullptr;
  fnBExecute execute = nullptr;
  bool loaded = false;
  std::string loadError;
};

CudnnApi &cudnnApiState() {
  static CudnnApi api;
  return api;
}

std::mutex &cudnnLoadMutex() {
  static std::mutex m;
  return m;
}

bool tryLoadFromDir(const std::string &dir, std::vector<void *> &handles) {
  auto openLib = [&](const char *soname) -> void * {
    std::string full = dir.empty() ? std::string(soname) : (dir + "/" + soname);
    void *h = dlopen(full.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (h) handles.push_back(h);
    return h;
  };
  void *ops = openLib("libcudnn_ops.so.9");
  void *graph = openLib("libcudnn_graph.so.9");
  void *shim = openLib("libcudnn.so.9");
  return (ops && graph) || shim;
}

std::string findNvidiaCudnnLibDir() {
  try {
    nb::gil_scoped_acquire gil;
    auto importlib = nb::module_::import_("importlib.util");
    nb::object spec = importlib.attr("find_spec")("nvidia.cudnn");
    if (!spec.is_none()) {
      nb::object locs = spec.attr("submodule_search_locations");
      if (!locs.is_none()) {
        for (nb::handle item : locs) {
          std::string base = nb::cast<std::string>(item);
          std::string libDir = base + "/lib";
          if (std::filesystem::exists(libDir + "/libcudnn.so.9") ||
              std::filesystem::exists(libDir + "/libcudnn_graph.so.9")) {
            return libDir;
          }
        }
      }
    }
  } catch (...) {
  }
  return {};
}

const CudnnApi &ensureCudnn() {
  auto &api = cudnnApiState();
  std::lock_guard<std::mutex> lock(cudnnLoadMutex());
  if (api.loaded) return api;
  api.loadError.clear();

  std::vector<void *> handles;
  if (!tryLoadFromDir("", handles)) {
    std::string wheelDir = findNvidiaCudnnLibDir();
    if (!wheelDir.empty()) {
      tryLoadFromDir(wheelDir, handles);
    }
  }

  auto lookup = [&](const char *name) -> void * {
    for (void *h : handles) {
      if (void *s = dlsym(h, name)) return s;
    }
    return dlsym(RTLD_DEFAULT, name);
  };

  auto loadSym = [&](auto &dst, const char *name) -> bool {
    void *sym = lookup(name);
    if (!sym) {
      api.loadError = std::string("libcudnn.so.9 missing symbol: ") + name;
      return false;
    }
    dst = reinterpret_cast<std::decay_t<decltype(dst)>>(sym);
    return true;
  };

  if (loadSym(api.create, "cudnnCreate") &&
      loadSym(api.destroy, "cudnnDestroy") &&
      loadSym(api.setStream, "cudnnSetStream") &&
      loadSym(api.createDesc, "cudnnBackendCreateDescriptor") &&
      loadSym(api.destroyDesc, "cudnnBackendDestroyDescriptor") &&
      loadSym(api.finalizeDesc, "cudnnBackendFinalize") &&
      loadSym(api.setAttr, "cudnnBackendSetAttribute") &&
      loadSym(api.getAttr, "cudnnBackendGetAttribute") &&
      loadSym(api.execute, "cudnnBackendExecute")) {
    api.getErrorString = reinterpret_cast<CudnnApi::fnGetErrorString>(lookup("cudnnGetErrorString"));
    api.getLastErrorString = reinterpret_cast<CudnnApi::fnGetLastErrorString>(lookup("cudnnGetLastErrorString"));
    api.loaded = true;
    return api;
  }

  if (api.loadError.empty())
    api.loadError = "libcudnn.so.9 / libcudnn_graph.so.9 not found";
  throw CUDAError(api.loadError);
}

void checkCudnn(cudnnStatus_t st, const char *what) {
  if (st == CUDNN_STATUS_SUCCESS) return;
  const auto &api = cudnnApiState();
  std::string msg = std::string(what) + " failed (status=" + std::to_string(st);
  if (api.getErrorString) {
    if (const char *s = api.getErrorString(st)) msg += std::string(": ") + s;
  }
  msg += ")";
  if (api.getLastErrorString) {
    char buf[512] = {};
    api.getLastErrorString(buf, sizeof(buf) - 1);
    if (buf[0]) msg += std::string(" - ") + buf;
  }
  throw CUDAError(msg);
}

void checkDriver(cuda_drv::CUresult rc, const char *what) {
  if (rc == cuda_drv::kCudaSuccess) return;
  throw CUDAError(std::string(what) + " failed with CUDA error " + std::to_string(rc));
}

struct Desc {
  cudnnBackendDescriptor_t ptr = nullptr;
  Desc() = default;
  explicit Desc(int type) {
    checkCudnn(ensureCudnn().createDesc(type, &ptr), "cudnnBackendCreateDescriptor");
  }
  ~Desc() { reset(); }
  Desc(const Desc &) = delete;
  Desc &operator=(const Desc &) = delete;
  Desc(Desc &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
  Desc &operator=(Desc &&other) noexcept {
    if (this != &other) {
      reset();
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }
  void reset() noexcept {
    if (ptr) {
      cudnnApiState().destroyDesc(ptr);
      ptr = nullptr;
    }
  }
  void set(int attr, int type, std::int64_t count, const void *data) const {
    checkCudnn(cudnnApiState().setAttr(ptr, attr, type, count, data), "cudnnBackendSetAttribute");
  }
  void finalize() const {
    checkCudnn(cudnnApiState().finalizeDesc(ptr), "cudnnBackendFinalize");
  }
};

std::vector<std::int64_t> contiguousStrides(const std::vector<std::int64_t> &dims) {
  std::vector<std::int64_t> s(dims.size(), 1);
  for (std::size_t i = dims.size(); i-- > 1;) {
    s[i - 1] = s[i] * dims[i];
  }
  return s;
}

int toCudnnDtype(const std::string &dtype) {
  if (dtype == "f16") return CUDNN_DATA_HALF;
  if (dtype == "bf16") return CUDNN_DATA_BFLOAT16;
  if (dtype == "f32") return CUDNN_DATA_FLOAT;
  if (dtype == "i32") return CUDNN_DATA_INT32;
  if (dtype == "i64") return CUDNN_DATA_INT64;
  throw CUDAError("unsupported cuDNN data type: " + dtype);
}

Desc makeTensorDesc(std::int64_t uid, int dtype,
                    const std::vector<std::int64_t> &dims,
                    const std::vector<std::int64_t> &strides,
                    bool isVirtual = false,
                    bool isByValue = false,
                    cudnnBackendDescriptor_t raggedDesc = nullptr) {
  Desc t(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  std::int64_t align = 16;
  std::uint8_t virt = isVirtual ? 1 : 0;
  std::uint8_t byVal = isByValue ? 1 : 0;
  t.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  t.set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype);
  t.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64,
        static_cast<std::int64_t>(dims.size()), dims.data());
  t.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
        static_cast<std::int64_t>(strides.size()), strides.data());
  t.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  if (isVirtual) t.set(CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &virt);
  if (isByValue) t.set(CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &byVal);
  if (raggedDesc) t.set(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raggedDesc);
  t.finalize();
  return t;
}

Desc makePointwiseOp(int mode, int mathPrec,
                     cudnnBackendDescriptor_t x,
                     cudnnBackendDescriptor_t b,
                     cudnnBackendDescriptor_t y,
                     cudnnBackendDescriptor_t t = nullptr,
                     std::int64_t axis = -1) {
  Desc pw(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
  pw.set(CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
  pw.set(CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &mathPrec);
  if (axis >= 0) pw.set(CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, 1, &axis);
  pw.finalize();

  Desc op(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
  op.set(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw.ptr);
  op.set(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
  if (b) op.set(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
  op.set(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
  if (t) op.set(CUDNN_ATTR_OPERATION_POINTWISE_TDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &t);
  op.finalize();
  return op;
}

bool tryBuildExecutionPlan(cudnnHandle_t handle,
                           const std::vector<cudnnBackendDescriptor_t> &ops,
                           Desc &outPlan,
                           std::size_t &outWsSize,
                           std::string &outErr) {
  const auto &api = ensureCudnn();
  Desc og(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  og.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  og.set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
         static_cast<std::int64_t>(ops.size()), ops.data());
  if (cudnnStatus_t st = api.finalizeDesc(og.ptr); st != CUDNN_STATUS_SUCCESS) {
    outErr = "opgraph finalize failed (" + std::to_string(st) + ")";
    return false;
  }

  Desc bestPlan;
  std::int64_t bestWs = std::numeric_limits<std::int64_t>::max();
  bool found = false;

  for (int mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_FALLBACK, CUDNN_HEUR_MODE_INSTANT}) {
    Desc heur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    if (api.setAttr(heur.ptr, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode) != CUDNN_STATUS_SUCCESS ||
        api.setAttr(heur.ptr, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &og.ptr) != CUDNN_STATUS_SUCCESS ||
        api.finalizeDesc(heur.ptr) != CUDNN_STATUS_SUCCESS) {
      continue;
    }

    std::int64_t count = 0;
    if (api.getAttr(heur.ptr, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr) != CUDNN_STATUS_SUCCESS ||
        count <= 0) {
      continue;
    }

    std::vector<Desc> cfgs;
    cfgs.reserve(static_cast<std::size_t>(count));
    std::vector<cudnnBackendDescriptor_t> cfgPtrs;
    cfgPtrs.reserve(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
      cfgs.emplace_back(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
      cfgPtrs.push_back(cfgs.back().ptr);
    }

    std::int64_t actual = 0;
    if (api.getAttr(heur.ptr, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    count, &actual, cfgPtrs.data()) != CUDNN_STATUS_SUCCESS || actual <= 0) {
      continue;
    }

    for (std::int64_t i = 0; i < actual; ++i) {
      Desc ep(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
      if (api.setAttr(ep.ptr, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle) != CUDNN_STATUS_SUCCESS ||
          api.setAttr(ep.ptr, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfgPtrs[i]) != CUDNN_STATUS_SUCCESS ||
          api.finalizeDesc(ep.ptr) != CUDNN_STATUS_SUCCESS) {
        continue;
      }
      std::int64_t ws = 0;
      std::int64_t wc = 0;
      if (api.getAttr(ep.ptr, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &wc, &ws) != CUDNN_STATUS_SUCCESS) {
        continue;
      }
      if (!found || ws < bestWs) {
        bestWs = ws;
        bestPlan = std::move(ep);
        found = true;
        if (ws == 0) break;
      }
    }
    if (found && bestWs == 0) break;
  }

  if (!found) {
    outErr = "no supported cuDNN execution plan found";
    return false;
  }
  outPlan = std::move(bestPlan);
  outWsSize = static_cast<std::size_t>(std::max<std::int64_t>(0, bestWs));
  return true;
}

Desc buildExecutionPlan(cudnnHandle_t handle,
                        const std::vector<cudnnBackendDescriptor_t> &ops,
                        std::size_t &outWsSize) {
  Desc plan;
  std::string err;
  if (!tryBuildExecutionPlan(handle, ops, plan, outWsSize, err)) {
    throw CUDAError("cuDNN plan build failed: " + err);
  }
  return plan;
}

void execVariantPack(cudnnHandle_t handle,
                     const Desc &plan,
                     const std::vector<std::int64_t> &uids,
                     const std::vector<void *> &ptrs,
                     std::uint64_t wsAddr) {
  Desc vp(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  vp.set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64,
         static_cast<std::int64_t>(uids.size()), uids.data());
  vp.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
         static_cast<std::int64_t>(ptrs.size()), ptrs.data());
  if (wsAddr != 0) {
    void *wsPtr = reinterpret_cast<void *>(wsAddr);
    vp.set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &wsPtr);
  }
  vp.finalize();
  checkCudnn(cudnnApiState().execute(handle, plan.ptr, vp.ptr), "cudnnBackendExecute");
}

struct MatmulPlan {
  Desc plan;
  std::size_t wsSize = 0;
  std::int64_t uidA = 1;
  std::int64_t uidB = 2;
  std::int64_t uidC = 3;
};

MatmulPlan buildMatmul3D(cudnnHandle_t handle,
                         std::int64_t batch, std::int64_t m, std::int64_t n, std::int64_t k,
                         int dtypeA, const std::vector<std::int64_t> &stridesA,
                         int dtypeB, const std::vector<std::int64_t> &stridesB,
                         int dtypeC, const std::vector<std::int64_t> &stridesC) {
  Desc ta = makeTensorDesc(1, dtypeA, {batch, m, k}, stridesA);
  Desc tb = makeTensorDesc(2, dtypeB, {batch, k, n}, stridesB);
  Desc tc = makeTensorDesc(3, dtypeC, {batch, m, n}, stridesC);
  Desc mm(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
  int compType = CUDNN_DATA_FLOAT;
  mm.set(CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compType);
  mm.finalize();
  Desc op(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
  op.set(CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ta.ptr);
  op.set(CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tb.ptr);
  op.set(CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tc.ptr);
  op.set(CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mm.ptr);
  op.finalize();
  MatmulPlan out;
  out.plan = buildExecutionPlan(handle, {op.ptr}, out.wsSize);
  return out;
}

void hashCombine(std::size_t &seed, std::size_t v) {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::uint64_t doubleToBits(double d) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &d, sizeof(bits));
  return bits;
}

void launchKernel1D(cuda_drv::CUfunction fn,
                    cuda_drv::CUstream stream,
                    std::int64_t totalThreads,
                    void **params) {
  if (totalThreads <= 0) return;
  unsigned block = 256;
  unsigned grid = static_cast<unsigned>((totalThreads + block - 1) / block);
  checkDriver(cuda_drv::driver().cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr),
              "cuLaunchKernel");
}

}  // namespace

struct CudnnPlan {
  std::string entry;
  std::string layout;
  bool isVarlen = false;
  bool isPadded = false;
  bool isBwdNative = false;
  bool isBwdHybrid = false;
  bool causal = false;
  bool wantOut = false;
  bool wantStats = false;
  bool sameKV = true;
  bool varlenDynamicMax = false;
  bool cuIsSeqlen = false;
  int dtypeCode = CUDNN_DATA_HALF;

  std::int64_t B = 0, H = 0, M = 0, N = 0, D = 0, Dv = 0;
  float scaleVal = 0.0f;
  float negInfVal = -std::numeric_limits<float>::infinity();

  Desc ep;
  std::size_t wsSize = 0;

  // Hybrid SM8x backward MatMul plans
  MatmulPlan mmS, mmDp, mmDv, mmDq, mmDk;
};

bool CudnnPlanKey::operator==(const CudnnPlanKey &other) const {
  return causal == other.causal &&
         scaleBits == other.scaleBits &&
         maxSq == other.maxSq &&
         maxSkv == other.maxSkv &&
         entry == other.entry &&
         layout == other.layout &&
         dtypes == other.dtypes &&
         shapes == other.shapes;
}

std::size_t CudnnPlanKeyHash::operator()(const CudnnPlanKey &k) const {
  std::size_t h = std::hash<std::string>{}(k.entry);
  hashCombine(h, std::hash<bool>{}(k.causal));
  hashCombine(h, std::hash<std::uint64_t>{}(k.scaleBits));
  hashCombine(h, std::hash<std::int64_t>{}(k.maxSq));
  hashCombine(h, std::hash<std::int64_t>{}(k.maxSkv));
  hashCombine(h, std::hash<std::string>{}(k.layout));
  for (const auto &dt : k.dtypes) hashCombine(h, std::hash<std::string>{}(dt));
  for (const auto &sh : k.shapes) {
    hashCombine(h, sh.size());
    for (auto d : sh) hashCombine(h, std::hash<std::int64_t>{}(d));
  }
  return h;
}

bool cudnnAvailable() {
  try {
    ensureCudnn();
    return true;
  } catch (...) {
    return false;
  }
}

CudnnSessionState::CudnnSessionState(std::shared_ptr<CUDASessionQueue> q)
    : queue(std::move(q)), session(queue ? queue->session : nullptr) {
  if (!session || session->closed())
    throw CUDAError("cuDNN operations require an open CUDA execution session");
  const auto &api = ensureCudnn();
  cuda_drv::ScopedCudaContext ctxGuard(queue->cudaCtx);
  checkCudnn(api.create(&cudnnHandle), "cudnnCreate");
  checkCudnn(api.setStream(cudnnHandle, reinterpret_cast<void *>(session->stream())), "cudnnSetStream");
  auto cap = session->capability();
  arch = cap[0] * 10 + cap[1];
}

CudnnSessionState::~CudnnSessionState() {
  close();
}

void CudnnSessionState::clearCache() {
  if (session && !session->closed() && (!plans.empty() || !scratchPool.empty())) {
    nb::gil_scoped_release release;
    session->synchronize();
  }
  plans.clear();
  scratchPool.clear();
}

void CudnnSessionState::close() {
  if (closed) return;
  closed = true;
  if (session && !session->closed() && (!plans.empty() || !scratchPool.empty() || aux.module)) {
    try {
      session->synchronize();
    } catch (...) {
    }
  }
  plans.clear();
  scratchPool.clear();
  if (aux.module && queue && queue->cudaCtx) {
    try {
      cuda_drv::ScopedCudaContext ctxGuard(queue->cudaCtx);
      cuda_drv::driver().cuModuleUnload(aux.module);
    } catch (...) {
    }
    aux = {};
  }
  if (cudnnHandle) {
    try {
      if (queue && queue->cudaCtx) {
        cuda_drv::ScopedCudaContext ctxGuard(queue->cudaCtx);
        cudnnApiState().destroy(cudnnHandle);
      } else {
        cudnnApiState().destroy(cudnnHandle);
      }
    } catch (...) {
    }
    cudnnHandle = nullptr;
  }
}

std::shared_ptr<DeviceBuffer> CudnnSessionState::scratch(std::size_t slot, std::size_t nbytes) {
  if (slot >= scratchPool.size()) scratchPool.resize(slot + 1);
  if (!scratchPool[slot] || scratchPool[slot]->size() < nbytes) {
    scratchPool[slot] = session->allocate(std::max<std::size_t>(nbytes, 16));
  }
  return scratchPool[slot];
}

const BwdAuxFunctions &CudnnSessionState::auxFunctions() {
  if (aux.module) return aux;
  const auto &drv = cuda_drv::driver();
  checkDriver(drv.cuModuleLoadData(&aux.module, kBwdAuxPtx), "cuModuleLoadData(cudnn_bwd_aux)");
  checkDriver(drv.cuModuleGetFunction(&aux.rowsumF16, aux.module, "bwd_rowsum_f16"), "cuModuleGetFunction(bwd_rowsum_f16)");
  checkDriver(drv.cuModuleGetFunction(&aux.rowsumBf16, aux.module, "bwd_rowsum_bf16"), "cuModuleGetFunction(bwd_rowsum_bf16)");
  checkDriver(drv.cuModuleGetFunction(&aux.softmaxScaleF16, aux.module, "bwd_softmax_scale_f16"), "cuModuleGetFunction(bwd_softmax_scale_f16)");
  checkDriver(drv.cuModuleGetFunction(&aux.softmaxScaleBf16, aux.module, "bwd_softmax_scale_bf16"), "cuModuleGetFunction(bwd_softmax_scale_bf16)");
  checkDriver(drv.cuModuleGetFunction(&aux.varlenMeta, aux.module, "varlen_meta_kernel"), "cuModuleGetFunction(varlen_meta_kernel)");
  checkDriver(drv.cuModuleGetFunction(&aux.varlenUnpack16, aux.module, "varlen_unpack_16"), "cuModuleGetFunction(varlen_unpack_16)");
  checkDriver(drv.cuModuleGetFunction(&aux.varlenPack16, aux.module, "varlen_pack_16"), "cuModuleGetFunction(varlen_pack_16)");
  return aux;
}

std::shared_ptr<CudnnPlan> CudnnSessionState::getOrCreatePlan(
    const std::string &entry,
    const std::vector<std::vector<std::int64_t>> &shapes,
    const std::vector<std::string> &dtypes,
    bool causal,
    bool hasScale,
    double scale,
    std::int64_t maxSq,
    std::int64_t maxSkv,
    const std::string &layout) {
  CudnnPlanKey key;
  key.entry = entry;
  key.shapes = shapes;
  key.dtypes = dtypes;
  key.causal = causal;
  key.scaleBits = hasScale ? doubleToBits(scale) : 0ULL;
  key.maxSq = maxSq;
  key.maxSkv = maxSkv;
  key.layout = layout;

  if (auto it = plans.find(key); it != plans.end()) {
    return it->second;
  }

  auto plan = std::make_shared<CudnnPlan>();
  plan->entry = entry;
  plan->causal = causal;
  plan->layout = layout;
  plan->isVarlen = (entry.find("varlen") != std::string::npos);
  plan->isPadded = (plan->isVarlen && shapes[0].size() == 4);
  plan->dtypeCode = toCudnnDtype(dtypes[0]);

  if (!plan->isVarlen) {
    plan->B = shapes[0][0];
    plan->H = shapes[0][1];
    plan->M = shapes[0][2];
    plan->D = shapes[0][3];
    plan->N = shapes[1][2];
    plan->Dv = shapes[2][3];
  } else if (plan->isPadded) {
    plan->B = shapes[0][0];
    plan->H = shapes[0][1];
    plan->M = shapes[0][2];
    plan->D = shapes[0][3];
    plan->N = shapes[1][2];
    plan->Dv = shapes[2][3];
    plan->cuIsSeqlen = (shapes[3].size() != 1 || shapes[3][0] == plan->B);
  } else {
    const auto &cuShape = (entry == "sdpa_varlen_bwd") ? shapes[6] : shapes[3];
    plan->B = cuShape[0] - 1;
    plan->H = shapes[0][1];
    plan->D = shapes[0][2];
    plan->Dv = shapes[2][2];
    plan->M = maxSq > 0 ? maxSq : 128;
    plan->N = maxSkv > 0 ? maxSkv : 128;
    plan->sameKV = (plan->D == plan->Dv);
  }
  plan->scaleVal = hasScale ? static_cast<float>(scale) : static_cast<float>(1.0 / std::sqrt(static_cast<double>(plan->D)));

  const std::int64_t B = plan->B, H = plan->H, M = plan->M, N = plan->N, D = plan->D, Dv = plan->Dv;

  if (entry == "sdpa_fwd" || entry == "sdpa_stats" ||
      entry == "sdpa_varlen_fwd" || entry == "sdpa_varlen_stats") {
    plan->wantOut = (entry == "sdpa_fwd" || entry == "sdpa_varlen_fwd");
    plan->wantStats = (entry == "sdpa_stats" || entry == "sdpa_varlen_stats");

    std::vector<Desc> hold;
    cudnnBackendDescriptor_t rqPtr = nullptr, rkPtr = nullptr, rvPtr = nullptr, slqPtr = nullptr, slkvPtr = nullptr;

    std::vector<std::int64_t> qStr, kStr, vStr, oStr;
    if (plan->isVarlen && !plan->isPadded) {
      hold.push_back(makeTensorDesc(301, CUDNN_DATA_INT64, {B + 1, 1, 1, 1}, {1, 1, 1, 1}));
      rqPtr = hold.back().ptr;
      hold.push_back(makeTensorDesc(302, CUDNN_DATA_INT64, {B + 1, 1, 1, 1}, {1, 1, 1, 1}));
      rkPtr = hold.back().ptr;
      if (!plan->sameKV) {
        hold.push_back(makeTensorDesc(311, CUDNN_DATA_INT64, {B + 1, 1, 1, 1}, {1, 1, 1, 1}));
        rvPtr = hold.back().ptr;
      } else {
        rvPtr = rkPtr;
      }
      qStr = {M * H * D, D, H * D, 1};
      kStr = {N * H * D, D, H * D, 1};
      vStr = {N * H * Dv, Dv, H * Dv, 1};
      oStr = {M * H * Dv, Dv, H * Dv, 1};
    } else {
      qStr = contiguousStrides({B, H, M, D});
      kStr = contiguousStrides({B, H, N, D});
      vStr = contiguousStrides({B, H, N, Dv});
      oStr = contiguousStrides({B, H, M, Dv});
    }
    if (plan->isVarlen) {
      hold.push_back(makeTensorDesc(303, CUDNN_DATA_INT32, {B, 1, 1, 1}, {1, 1, 1, 1}));
      slqPtr = hold.back().ptr;
      hold.push_back(makeTensorDesc(304, CUDNN_DATA_INT32, {B, 1, 1, 1}, {1, 1, 1, 1}));
      slkvPtr = hold.back().ptr;
    }

    Desc qd = makeTensorDesc(305, plan->dtypeCode, {B, H, M, D}, qStr, false, false, rqPtr);
    Desc kd = makeTensorDesc(306, plan->dtypeCode, {B, H, N, D}, kStr, false, false, rkPtr);
    Desc vd = makeTensorDesc(307, plan->dtypeCode, {B, H, N, Dv}, vStr, false, false, rvPtr);
    Desc od = makeTensorDesc(308, plan->dtypeCode, {B, H, M, Dv}, oStr, !plan->wantOut, false, rqPtr);
    Desc scd = makeTensorDesc(310, CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, false, true);
    Desc stdDesc;
    if (plan->wantStats) {
      stdDesc = makeTensorDesc(309, CUDNN_DATA_FLOAT, {B, H, M, 1}, {H * M, M, 1, 1});
    }

    Desc op(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &qd.ptr);
    op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &kd.ptr);
    op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &vd.ptr);
    op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &od.ptr);
    if (plan->wantStats) {
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &stdDesc.ptr);
    }
    op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scd.ptr);
    if (plan->isVarlen) {
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &slqPtr);
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &slkvPtr);
    }

    if (causal) {
      std::vector<std::int64_t> maskDims = {B, H, M, N};
      std::vector<std::int64_t> maskStr = {H * M * N, M * N, N, 1};
      Desc sgIn = makeTensorDesc(400, CUDNN_DATA_FLOAT, maskDims, maskStr, true);
      Desc sgRow = makeTensorDesc(401, CUDNN_DATA_INT32, maskDims, maskStr, true);
      Desc sgCol = makeTensorDesc(402, CUDNN_DATA_INT32, maskDims, maskStr, true);
      Desc sgMask = makeTensorDesc(403, CUDNN_DATA_BOOLEAN, maskDims, maskStr, true);
      Desc sgNegInf = makeTensorDesc(404, CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, false, true);
      Desc sgOut = makeTensorDesc(405, CUDNN_DATA_FLOAT, maskDims, maskStr, true);

      Desc pw0 = makePointwiseOp(CUDNN_POINTWISE_GEN_INDEX, CUDNN_DATA_INT32, sgIn.ptr, nullptr, sgRow.ptr, nullptr, 2);
      Desc pw1 = makePointwiseOp(CUDNN_POINTWISE_GEN_INDEX, CUDNN_DATA_INT32, sgIn.ptr, nullptr, sgCol.ptr, nullptr, 3);
      Desc pw2 = makePointwiseOp(CUDNN_POINTWISE_CMP_GE, CUDNN_DATA_INT32, sgRow.ptr, sgCol.ptr, sgMask.ptr);
      Desc pw3 = makePointwiseOp(CUDNN_POINTWISE_BINARY_SELECT, CUDNN_DATA_FLOAT, sgIn.ptr, sgNegInf.ptr, sgOut.ptr, sgMask.ptr);

      std::vector<cudnnBackendDescriptor_t> sgOps = {pw0.ptr, pw1.ptr, pw2.ptr, pw3.ptr};
      Desc sg(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
      sg.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle);
      sg.set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
             static_cast<std::int64_t>(sgOps.size()), sgOps.data());
      sg.finalize();

      std::int64_t inUid = 400, outUid = 405;
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &sg.ptr);
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID, CUDNN_TYPE_INT64, 1, &inUid);
      op.set(CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID, CUDNN_TYPE_INT64, 1, &outUid);
      op.finalize();
      plan->ep = buildExecutionPlan(cudnnHandle, {op.ptr}, plan->wsSize);
    } else {
      op.finalize();
      plan->ep = buildExecutionPlan(cudnnHandle, {op.ptr}, plan->wsSize);
    }

    plans.emplace(key, plan);
    return plan;
  }

  if (entry == "sdpa_bwd" || entry == "sdpa_varlen_bwd") {
    // Try native cuDNN op 43 on Hopper+ non-causal/non-varlen first (or whenever supported),
    // cleanly falling back to verified 5x 3D MatMul + SM80 PTX backward.
    if (arch >= 90 && !op43Unsupported && !causal && !plan->isVarlen) {
      try {
        Desc qd = makeTensorDesc(305, plan->dtypeCode, {B, H, M, D}, contiguousStrides({B, H, M, D}));
        Desc kd = makeTensorDesc(306, plan->dtypeCode, {B, H, N, D}, contiguousStrides({B, H, N, D}));
        Desc vd = makeTensorDesc(307, plan->dtypeCode, {B, H, N, Dv}, contiguousStrides({B, H, N, Dv}));
        Desc od = makeTensorDesc(308, plan->dtypeCode, {B, H, M, Dv}, contiguousStrides({B, H, M, Dv}));
        Desc dod = makeTensorDesc(312, plan->dtypeCode, {B, H, M, Dv}, contiguousStrides({B, H, M, Dv}));
        Desc stdDesc = makeTensorDesc(309, CUDNN_DATA_FLOAT, {B, H, M, 1}, {H * M, M, 1, 1});
        Desc scd = makeTensorDesc(310, CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, false, true);
        Desc dqd = makeTensorDesc(313, plan->dtypeCode, {B, H, M, D}, contiguousStrides({B, H, M, D}));
        Desc dkd = makeTensorDesc(314, plan->dtypeCode, {B, H, N, D}, contiguousStrides({B, H, N, D}));
        Desc dvd = makeTensorDesc(315, plan->dtypeCode, {B, H, N, Dv}, contiguousStrides({B, H, N, Dv}));

        Desc op(CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &qd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &kd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &vd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &od.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dod.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &stdDesc.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dqd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dkd.ptr);
        op.set(CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dvd.ptr);
        Desc epNative;
        std::size_t ws = 0;
        std::string err;
        if (cudnnApiState().finalizeDesc(op.ptr) == CUDNN_STATUS_SUCCESS &&
            tryBuildExecutionPlan(cudnnHandle, {op.ptr}, epNative, ws, err)) {
          plan->isBwdNative = true;
          plan->ep = std::move(epNative);
          plan->wsSize = ws;
          plans.emplace(key, plan);
          return plan;
        }
        op43Unsupported = true;
      } catch (...) {
        op43Unsupported = true;
      }
    }

    plan->isBwdHybrid = true;
    const std::int64_t BH = B * H;
    const int dt = plan->dtypeCode;
    plan->mmS = buildMatmul3D(cudnnHandle, BH, M, N, D,
                              dt, {M * D, D, 1}, dt, {N * D, 1, D}, CUDNN_DATA_FLOAT, {M * N, N, 1});
    plan->mmDp = buildMatmul3D(cudnnHandle, BH, M, N, Dv,
                               dt, {M * Dv, Dv, 1}, dt, {N * Dv, 1, Dv}, CUDNN_DATA_FLOAT, {M * N, N, 1});
    plan->mmDv = buildMatmul3D(cudnnHandle, BH, N, Dv, M,
                               dt, {M * N, 1, N}, dt, {M * Dv, Dv, 1}, dt, {N * Dv, Dv, 1});
    plan->mmDq = buildMatmul3D(cudnnHandle, BH, M, D, N,
                               dt, {M * N, N, 1}, dt, {N * D, D, 1}, dt, {M * D, D, 1});
    plan->mmDk = buildMatmul3D(cudnnHandle, BH, N, D, M,
                               dt, {M * N, 1, N}, dt, {M * D, D, 1}, dt, {N * D, D, 1});
    plan->wsSize = std::max({plan->mmS.wsSize, plan->mmDp.wsSize, plan->mmDv.wsSize,
                             plan->mmDq.wsSize, plan->mmDk.wsSize});
    plans.emplace(key, plan);
    return plan;
  }

  throw CUDAError("unknown cuDNN entry: " + entry);
}

CudnnExecutable::CudnnExecutable(std::shared_ptr<CudnnSessionState> stateIn,
                                 nb::object sessionObjIn,
                                 nb::object tensorTypeIn,
                                 std::string entryIn,
                                 std::vector<OutputSpec> outputSpecsIn,
                                 bool causalIn,
                                 bool hasScaleIn,
                                 double scaleIn,
                                 std::int64_t maxSqIn,
                                 std::int64_t maxSkvIn,
                                 std::string layoutIn,
                                 nb::object artifactDirIn,
                                 std::string opIdIn)
    : state(std::move(stateIn)),
      sessionObj(std::move(sessionObjIn)),
      tensorType(std::move(tensorTypeIn)),
      entry(std::move(entryIn)),
      outputSpecs(std::move(outputSpecsIn)),
      causal(causalIn),
      hasScale(hasScaleIn),
      scale(scaleIn),
      maxSq(maxSqIn),
      maxSkv(maxSkvIn),
      layout(std::move(layoutIn)),
      opId(std::move(opIdIn)),
      sourceStr("cudnn::" + entry),
      sourcePy(sourceStr.c_str()),
      artifactDir(std::move(artifactDirIn)) {
  reportDict["compiler"] = "cudnn";
  reportDict["entry"] = entry;
  nb::dict step;
  step["id"] = opId;
  step["kind"] = "external";
  nb::list steps;
  steps.append(step);
  executionPlanDict["steps"] = steps;
}

void CudnnExecutable::executeRaw(const std::vector<DeviceBuffer *> &inBufs,
                                 const std::vector<std::vector<std::int64_t>> &inShapes,
                                 const std::vector<std::string> &inDtypes,
                                 std::vector<std::shared_ptr<DeviceBuffer>> &outBufs,
                                 std::vector<std::vector<std::int64_t>> &outShapes,
                                 std::vector<std::string> &outDtypes) {
  if (!state || state->closed || state->session->closed())
    throw CUDAError("cuDNN session is closed");

  cuda_drv::ScopedCudaContext ctxGuard(state->queue->cudaCtx);
  const auto stream = reinterpret_cast<cuda_drv::CUstream>(state->session->stream());
  const auto &drv = cuda_drv::driver();

  std::int64_t effMaxSq = maxSq;
  std::int64_t effMaxSkv = maxSkv;
  bool isVarlen = (entry.find("varlen") != std::string::npos);
  bool isPadded = (isVarlen && inShapes[0].size() == 4);
  std::size_t cuQIdx = (entry == "sdpa_varlen_bwd") ? 6 : 3;
  std::size_t cuKvIdx = (entry == "sdpa_varlen_bwd") ? 7 : 4;

  if (isVarlen && isPadded) {
    if (effMaxSq <= 0) effMaxSq = inShapes[0][2];
    if (effMaxSkv <= 0) effMaxSkv = inShapes[1][2];
  } else if (isVarlen && (effMaxSq <= 0 || effMaxSkv <= 0)) {
    auto maxLenFromCu = [&](const DeviceBuffer *buf) -> std::int64_t {
      std::size_t n = buf->size() / sizeof(std::int32_t);
      if (n <= 1) return 1;
      std::vector<std::int32_t> host(n);
      checkDriver(drv.cuMemcpyDtoHAsync(host.data(), buf->pointer(), buf->size(), stream), "cuMemcpyDtoHAsync");
      checkDriver(drv.cuStreamSynchronize(stream), "cuStreamSynchronize");
      std::int32_t mx = 1;
      for (std::size_t i = 0; i + 1 < n; ++i) mx = std::max(mx, host[i + 1] - host[i]);
      return mx;
    };
    if (effMaxSq <= 0 && cuQIdx < inBufs.size()) effMaxSq = maxLenFromCu(inBufs[cuQIdx]);
    if (effMaxSkv <= 0 && cuKvIdx < inBufs.size()) effMaxSkv = maxLenFromCu(inBufs[cuKvIdx]);
  }

  auto plan = state->getOrCreatePlan(entry, inShapes, inDtypes, causal, hasScale, scale,
                                     effMaxSq, effMaxSkv, layout);

  std::uint64_t wsAddr = 0;
  if (plan->wsSize > 0) {
    wsAddr = state->scratch(0, plan->wsSize)->pointer();
  }

  auto cuPtr = [](const DeviceBuffer *b) -> void * {
    return reinterpret_cast<void *>(b->pointer());
  };

  const std::int64_t B = plan->B, H = plan->H, M = plan->M, N = plan->N, D = plan->D, Dv = plan->Dv;
  std::uint64_t roQAddr = 0, roKAddr = 0, roVAddr = 0, slQAddr = 0, slKvAddr = 0;

  if (plan->isVarlen) {
    const auto &aux = state->auxFunctions();
    if (plan->cuIsSeqlen) {
      slQAddr = inBufs[cuQIdx]->pointer();
      slKvAddr = inBufs[cuKvIdx]->pointer();
    } else {
      auto bRoQ = state->scratch(1, static_cast<std::size_t>(B + 1) * sizeof(std::int64_t));
      auto bRoK = state->scratch(2, static_cast<std::size_t>(B + 1) * sizeof(std::int64_t));
      auto bRoV = state->scratch(3, static_cast<std::size_t>(B + 1) * sizeof(std::int64_t));
      auto bSlQ = state->scratch(4, static_cast<std::size_t>(B) * sizeof(std::int32_t));
      auto bSlKv = state->scratch(5, static_cast<std::size_t>(B) * sizeof(std::int32_t));
      roQAddr = bRoQ->pointer();
      roKAddr = bRoK->pointer();
      roVAddr = bRoV->pointer();
      slQAddr = bSlQ->pointer();
      slKvAddr = bSlKv->pointer();
      std::uint64_t cuQPtr = inBufs[cuQIdx]->pointer();
      std::uint64_t cuKvPtr = inBufs[cuKvIdx]->pointer();
      std::int32_t bI32 = static_cast<std::int32_t>(B);
      std::int64_t strQ = H * D, strK = H * D, strV = H * Dv;
      void *args[] = {&cuQPtr, &cuKvPtr, &roQAddr, &roKAddr, &roVAddr, &slQAddr, &slKvAddr,
                      &bI32, &strQ, &strK, &strV};
      launchKernel1D(aux.varlenMeta, stream, B + 1, args);
    }
  }

  outBufs.clear();
  outShapes.clear();
  outDtypes.clear();

  if (entry == "sdpa_fwd" || entry == "sdpa_stats") {
    bool isFwd = (entry == "sdpa_fwd");
    std::vector<std::int64_t> outShape = isFwd ? std::vector<std::int64_t>{B, H, M, Dv}
                                               : std::vector<std::int64_t>{B, H, M, 1};
    std::string outDt = isFwd ? inDtypes[0] : "f32";
    auto outBuf = state->session->allocate(
        static_cast<std::size_t>(B * H * M * (isFwd ? Dv : 1)) * elemByteWidth(outDt));

    std::vector<std::int64_t> uids = {305, 306, 307, isFwd ? 308 : 309, 310};
    std::vector<void *> ptrs = {cuPtr(inBufs[0]), cuPtr(inBufs[1]), cuPtr(inBufs[2]),
                                cuPtr(outBuf.get()), &plan->scaleVal};
    if (causal) {
      uids.push_back(404);
      ptrs.push_back(&plan->negInfVal);
    }
    execVariantPack(state->cudnnHandle, plan->ep, uids, ptrs, wsAddr);
    outBufs.push_back(std::move(outBuf));
    outShapes.push_back(std::move(outShape));
    outDtypes.push_back(std::move(outDt));
    return;
  }

  if (entry == "sdpa_varlen_fwd" || entry == "sdpa_varlen_stats") {
    bool isFwd = (entry == "sdpa_varlen_fwd");
    std::vector<std::int64_t> outShape = isFwd ? inShapes[0] : std::vector<std::int64_t>{B, H, M, 1};
    if (isFwd) outShape.back() = Dv;
    std::string outDt = isFwd ? inDtypes[0] : "f32";
    std::size_t elems = 1;
    for (auto d : outShape) elems *= static_cast<std::size_t>(d);
    auto outBuf = state->session->allocate(elems * elemByteWidth(outDt));

    std::vector<std::int64_t> uids;
    std::vector<void *> ptrs;
    if (!plan->isPadded) {
      uids.push_back(301);
      ptrs.push_back(reinterpret_cast<void *>(roQAddr));
      uids.push_back(302);
      ptrs.push_back(reinterpret_cast<void *>(roKAddr));
      if (!plan->sameKV) {
        uids.push_back(311);
        ptrs.push_back(reinterpret_cast<void *>(roVAddr));
      }
    }
    uids.insert(uids.end(), {303, 304, 305, 306, 307, isFwd ? 308 : 309, 310});
    ptrs.insert(ptrs.end(), {reinterpret_cast<void *>(slQAddr), reinterpret_cast<void *>(slKvAddr),
                             cuPtr(inBufs[0]), cuPtr(inBufs[1]), cuPtr(inBufs[2]),
                             cuPtr(outBuf.get()), &plan->scaleVal});
    if (causal) {
      uids.push_back(404);
      ptrs.push_back(&plan->negInfVal);
    }
    execVariantPack(state->cudnnHandle, plan->ep, uids, ptrs, wsAddr);
    outBufs.push_back(std::move(outBuf));
    outShapes.push_back(std::move(outShape));
    outDtypes.push_back(std::move(outDt));
    return;
  }

  // Backward: sdpa_bwd or sdpa_varlen_bwd
  std::size_t qBytes = inBufs[0]->size();
  std::size_t kBytes = inBufs[1]->size();
  std::size_t vBytes = inBufs[2]->size();
  auto dqBuf = state->session->allocate(qBytes);
  auto dkBuf = state->session->allocate(kBytes);
  auto dvBuf = state->session->allocate(vBytes);

  if (plan->isBwdNative) {
    std::vector<std::int64_t> uids = {305, 306, 307, 308, 312, 309, 310, 313, 314, 315};
    std::vector<void *> ptrs = {cuPtr(inBufs[0]), cuPtr(inBufs[1]), cuPtr(inBufs[2]),
                                cuPtr(inBufs[3]), cuPtr(inBufs[4]), cuPtr(inBufs[5]),
                                &plan->scaleVal, cuPtr(dqBuf.get()), cuPtr(dkBuf.get()), cuPtr(dvBuf.get())};
    execVariantPack(state->cudnnHandle, plan->ep, uids, ptrs, wsAddr);
    outBufs = {std::move(dqBuf), std::move(dkBuf), std::move(dvBuf)};
    outShapes = {inShapes[0], inShapes[1], inShapes[2]};
    outDtypes = {inDtypes[0], inDtypes[1], inDtypes[2]};
    return;
  }

  const auto &aux = state->auxFunctions();
  const std::int64_t BH = B * H;
  const bool needUnpack = (plan->isVarlen && !plan->isPadded);

  std::uint64_t qPtr = inBufs[0]->pointer();
  std::uint64_t kPtr = inBufs[1]->pointer();
  std::uint64_t vPtr = inBufs[2]->pointer();
  std::uint64_t oPtr = inBufs[3]->pointer();
  std::uint64_t doPtr = inBufs[4]->pointer();
  std::uint64_t statsPtr = inBufs[5]->pointer();
  std::uint64_t dqWorkPtr = dqBuf->pointer();
  std::uint64_t dkWorkPtr = dkBuf->pointer();
  std::uint64_t dvWorkPtr = dvBuf->pointer();

  if (needUnpack) {
    std::size_t padQ = static_cast<std::size_t>(BH * M * D) * 2;
    std::size_t padK = static_cast<std::size_t>(BH * N * D) * 2;
    std::size_t padV = static_cast<std::size_t>(BH * N * Dv) * 2;
    std::size_t padO = static_cast<std::size_t>(BH * M * Dv) * 2;
    auto pq = state->scratch(11, padQ);
    auto pk = state->scratch(12, padK);
    auto pv = state->scratch(13, padV);
    auto po = state->scratch(14, padO);
    auto pdo = state->scratch(15, padO);
    auto pdq = state->scratch(16, padQ);
    auto pdk = state->scratch(17, padK);
    auto pdv = state->scratch(18, padV);

    std::uint64_t cuQPtr = inBufs[6]->pointer();
    std::uint64_t cuKvPtr = inBufs[7]->pointer();
    std::int32_t bI32 = static_cast<std::int32_t>(B);
    std::int64_t h64 = H, m64 = M, n64 = N, d64 = D, dv64 = Dv;

    auto launchUnpack = [&](std::uint64_t src, std::uint64_t cu, std::uint64_t dst, std::int64_t seq, std::int64_t feat) {
      void *args[] = {&src, &cu, &dst, &bI32, &h64, &seq, &feat};
      launchKernel1D(aux.varlenUnpack16, stream, BH * seq * feat, args);
    };
    launchUnpack(qPtr, cuQPtr, pq->pointer(), m64, d64);
    launchUnpack(kPtr, cuKvPtr, pk->pointer(), n64, d64);
    launchUnpack(vPtr, cuKvPtr, pv->pointer(), n64, dv64);
    launchUnpack(oPtr, cuQPtr, po->pointer(), m64, dv64);
    launchUnpack(doPtr, cuQPtr, pdo->pointer(), m64, dv64);

    qPtr = pq->pointer();
    kPtr = pk->pointer();
    vPtr = pv->pointer();
    oPtr = po->pointer();
    doPtr = pdo->pointer();
    dqWorkPtr = pdq->pointer();
    dkWorkPtr = pdk->pointer();
    dvWorkPtr = pdv->pointer();
  }

  auto dSumBuf = state->scratch(6, static_cast<std::size_t>(BH * M) * sizeof(float));
  auto sBuf = state->scratch(7, static_cast<std::size_t>(BH * M * N) * sizeof(float));
  auto dpBuf = state->scratch(8, static_cast<std::size_t>(BH * M * N) * sizeof(float));
  auto p16Buf = state->scratch(9, static_cast<std::size_t>(BH * M * N) * 2);
  auto ds16Buf = state->scratch(10, static_cast<std::size_t>(BH * M * N) * 2);

  std::uint64_t dSumPtr = dSumBuf->pointer();
  std::uint64_t sPtr = sBuf->pointer();
  std::uint64_t dpPtr = dpBuf->pointer();
  std::uint64_t p16Ptr = p16Buf->pointer();
  std::uint64_t ds16Ptr = ds16Buf->pointer();

  std::int64_t bh64 = BH, h64 = H, m64 = M, n64 = N, dv64 = Dv;
  bool isBf16 = (inDtypes[0] == "bf16");
  {
    void *args[] = {&doPtr, &oPtr, &dSumPtr, &bh64, &m64, &dv64};
    launchKernel1D(isBf16 ? aux.rowsumBf16 : aux.rowsumF16, stream, BH * M, args);
  }

  auto voidAddr = [](std::uint64_t addr) -> void * { return reinterpret_cast<void *>(addr); };
  execVariantPack(state->cudnnHandle, plan->mmS.plan, {1, 2, 3},
                  {voidAddr(qPtr), voidAddr(kPtr), voidAddr(sPtr)}, wsAddr);
  execVariantPack(state->cudnnHandle, plan->mmDp.plan, {1, 2, 3},
                  {voidAddr(doPtr), voidAddr(vPtr), voidAddr(dpPtr)}, wsAddr);

  {
    std::int32_t causalI32 = causal ? 1 : 0;
    float scF32 = plan->scaleVal;
    void *args[] = {&sPtr, &dpPtr, &statsPtr, &dSumPtr, &slQAddr, &slKvAddr,
                    &p16Ptr, &ds16Ptr, &bh64, &h64, &m64, &n64, &scF32, &causalI32};
    launchKernel1D(isBf16 ? aux.softmaxScaleBf16 : aux.softmaxScaleF16, stream, BH * M * N, args);
  }

  execVariantPack(state->cudnnHandle, plan->mmDv.plan, {1, 2, 3},
                  {voidAddr(p16Ptr), voidAddr(doPtr), voidAddr(dvWorkPtr)}, wsAddr);
  execVariantPack(state->cudnnHandle, plan->mmDq.plan, {1, 2, 3},
                  {voidAddr(ds16Ptr), voidAddr(kPtr), voidAddr(dqWorkPtr)}, wsAddr);
  execVariantPack(state->cudnnHandle, plan->mmDk.plan, {1, 2, 3},
                  {voidAddr(ds16Ptr), voidAddr(qPtr), voidAddr(dkWorkPtr)}, wsAddr);

  if (needUnpack) {
    checkDriver(drv.cuMemsetD8Async(dqBuf->pointer(), 0, qBytes, stream), "cuMemsetD8Async");
    checkDriver(drv.cuMemsetD8Async(dkBuf->pointer(), 0, kBytes, stream), "cuMemsetD8Async");
    checkDriver(drv.cuMemsetD8Async(dvBuf->pointer(), 0, vBytes, stream), "cuMemsetD8Async");

    std::uint64_t cuQPtr = inBufs[6]->pointer();
    std::uint64_t cuKvPtr = inBufs[7]->pointer();
    std::int32_t bI32 = static_cast<std::int32_t>(B);
    std::int64_t d64 = D;
    auto launchPack = [&](std::uint64_t src, std::uint64_t cu, std::uint64_t dst, std::int64_t seq, std::int64_t feat) {
      void *args[] = {&src, &cu, &dst, &bI32, &h64, &seq, &feat};
      launchKernel1D(aux.varlenPack16, stream, BH * seq * feat, args);
    };
    launchPack(dqWorkPtr, cuQPtr, dqBuf->pointer(), m64, d64);
    launchPack(dkWorkPtr, cuKvPtr, dkBuf->pointer(), n64, d64);
    launchPack(dvWorkPtr, cuKvPtr, dvBuf->pointer(), n64, dv64);
  }

  outBufs = {std::move(dqBuf), std::move(dkBuf), std::move(dvBuf)};
  outShapes = {inShapes[0], inShapes[1], inShapes[2]};
  outDtypes = {inDtypes[0], inDtypes[1], inDtypes[2]};
}

nb::list CudnnExecutable::run(nb::handle inputsObj) {
  auto *rt = state->queue.get();
  nb::tuple tensors;
  std::vector<NativeDeviceTensor *> nativeTensors;
  bool fast = extractInputsForSession(inputsObj, tensorType, sessionObj, rt, tensors, nativeTensors);

  std::vector<std::shared_ptr<DeviceBuffer>> pinnedBufs;
  std::vector<DeviceBuffer *> rawBufs;
  std::vector<std::vector<std::int64_t>> inShapes;
  std::vector<std::string> inDtypes;

  if (fast) {
    if (rt->session->closed())
      throw CUDAError("CUDA execution session is closed");
    prepareAndValidateExtracted(nativeTensors, tensors, sessionObj, rt);
    rt->reapIfFull(64);
    pinnedBufs.reserve(nativeTensors.size());
    rawBufs.reserve(nativeTensors.size());
    inShapes.reserve(nativeTensors.size());
    inDtypes.reserve(nativeTensors.size());
    for (auto *nd : nativeTensors) {
      pinnedBufs.push_back(nd->nativeBuffer);
      rawBufs.push_back(nd->nativeBuffer.get());
      inShapes.push_back(nd->shapeVec);
      inDtypes.push_back(nd->dtypeStr);
    }
  } else {
    tensors = nb::tuple(inputsObj);
    for (nb::handle t : tensors) {
      if (!t.attr("_owner").is_none()) t.attr("_prepare_access")(sessionObj);
      auto buf = nb::cast<std::shared_ptr<DeviceBuffer>>(t.attr("_buffer"));
      pinnedBufs.push_back(buf);
      rawBufs.push_back(buf.get());
      inShapes.push_back(nb::cast<std::vector<std::int64_t>>(t.attr("shape")));
      inDtypes.push_back(nb::cast<std::string>(t.attr("dtype")));
    }
  }

  std::vector<std::shared_ptr<DeviceBuffer>> outBufs;
  std::vector<std::vector<std::int64_t>> outShapes;
  std::vector<std::string> outDtypes;
  {
    nb::gil_scoped_release release;
    executeRaw(rawBufs, inShapes, inDtypes, outBufs, outShapes, outDtypes);
  }

  if (fast) {
    rt->retainStreamWork(tensors);
  }

  nb::list wrapped;
  for (std::size_t i = 0; i < outBufs.size(); ++i) {
    if (fast) {
      if (i < outputSpecs.size() && outShapes[i] == outputSpecs[i].shape) {
        wrapped.append(NativeDeviceTensor::makeSubclass(
            tensorType, rt->sessionWeakRef, std::move(outBufs[i]),
            outputSpecs[i].shapeTuple, outputSpecs[i].dtypePyStr,
            std::move(outShapes[i]), std::move(outDtypes[i]),
            rt->sessionPtr, rt->descriptorStream));
      } else {
        nb::tuple shTuple(nb::cast(outShapes[i]));
        nb::str dtPy(outDtypes[i].c_str());
        wrapped.append(NativeDeviceTensor::makeSubclass(
            tensorType, rt->sessionWeakRef, std::move(outBufs[i]),
            std::move(shTuple), std::move(dtPy),
            std::move(outShapes[i]), std::move(outDtypes[i]),
            rt->sessionPtr, rt->descriptorStream));
      }
    } else {
      nb::tuple shTuple(nb::cast(outShapes[i]));
      nb::str dtPy(outDtypes[i].c_str());
      wrapped.append(tensorType(sessionObj, nb::cast(std::move(outBufs[i])), shTuple, dtPy));
    }
  }
  return wrapped;
}

CudnnCompositeExecutable::CudnnCompositeExecutable(
    std::shared_ptr<CudnnSessionState> stateIn,
    nb::object sessionObjIn,
    nb::object tensorTypeIn,
    std::size_t numSlotsIn,
    std::vector<std::size_t> initialInputSlotsIn,
    std::vector<std::size_t> finalOutputSlotsIn,
    std::vector<TensorPort> initialInputPortsIn,
    std::vector<TensorPort> finalOutputPortsIn,
    std::vector<std::vector<std::int64_t>> slotShapesIn,
    std::vector<std::string> slotDtypesIn,
    std::vector<Stage> stagesIn,
    nb::object artifactDirIn,
    nb::dict executionPlanIn,
    nb::dict reportIn)
    : state(std::move(stateIn)),
      sessionObj(std::move(sessionObjIn)),
      tensorType(std::move(tensorTypeIn)),
      numSlots(numSlotsIn),
      initialInputSlots(std::move(initialInputSlotsIn)),
      finalOutputSlots(std::move(finalOutputSlotsIn)),
      initialInputPorts(std::move(initialInputPortsIn)),
      finalOutputPorts(std::move(finalOutputPortsIn)),
      slotShapes(std::move(slotShapesIn)),
      slotDtypes(std::move(slotDtypesIn)),
      stages(std::move(stagesIn)),
      artifactDir(std::move(artifactDirIn)),
      executionPlanDict(std::move(executionPlanIn)),
      reportDict(std::move(reportIn)),
      sourcePy("composite") {}

nb::list CudnnCompositeExecutable::run(nb::handle inputsObj) {
  auto *rt = state->queue.get();
  nb::tuple tensors;
  std::vector<NativeDeviceTensor *> nativeTensors;
  bool fast = extractInputsForSession(inputsObj, tensorType, sessionObj, rt, tensors, nativeTensors);

  if (fast) {
    if (rt->session->closed())
      throw CUDAError("CUDA execution session is closed");
    if (nativeTensors.size() != initialInputPorts.size())
      throw CUDAError("program expects " + std::to_string(initialInputPorts.size()) + " inputs");
    for (std::size_t i = 0; i < initialInputPorts.size(); ++i) {
      if (nativeTensors[i]->shapeVec != initialInputPorts[i].shapeVec ||
          nativeTensors[i]->dtypeStr != initialInputPorts[i].dtypeStr)
        throw CUDAError("program port shape/dtype differs from its compiled type");
    }
    prepareAndValidateExtracted(nativeTensors, tensors, sessionObj, rt);
    rt->reapIfFull(64);
  } else {
    tensors = nb::tuple(inputsObj);
    if (tensors.size() != initialInputPorts.size())
      throw CUDAError("program expects " + std::to_string(initialInputPorts.size()) + " inputs");
    for (nb::handle t : tensors) {
      if (!t.attr("_owner").is_none()) t.attr("_prepare_access")(sessionObj);
    }
  }

  std::vector<std::shared_ptr<DeviceBuffer>> slotBufs(numSlots);
  std::vector<std::shared_ptr<DeviceBuffer>> initialBufs;
  initialBufs.reserve(tensors.size());
  for (std::size_t i = 0; i < initialInputSlots.size(); ++i) {
    auto buf = fast ? nativeTensors[i]->nativeBuffer
                    : nb::cast<std::shared_ptr<DeviceBuffer>>(tensors[i].attr("_buffer"));
    slotBufs[initialInputSlots[i]] = buf;
    initialBufs.push_back(std::move(buf));
  }

  std::shared_ptr<Event> lastAskEvent;
  std::vector<std::shared_ptr<DeviceBuffer>> intermediateBufs;
  {
    nb::gil_scoped_release release;
    for (const auto &stage : stages) {
      if (!stage.isExternal) {
        std::vector<std::shared_ptr<DeviceBuffer>> stageIns;
        stageIns.reserve(stage.inputSlots.size());
        for (auto slot : stage.inputSlots) stageIns.push_back(slotBufs[slot]);
        auto res = stage.program->run(stageIns);
        lastAskEvent = std::move(res.second);
        for (std::size_t k = 0; k < stage.outputSlots.size(); ++k) {
          slotBufs[stage.outputSlots[k]] = res.first[k];
          intermediateBufs.push_back(std::move(res.first[k]));
        }
      } else {
        std::vector<DeviceBuffer *> inPtrs;
        std::vector<std::vector<std::int64_t>> inShapes;
        std::vector<std::string> inDtypes;
        inPtrs.reserve(stage.inputSlots.size());
        inShapes.reserve(stage.inputSlots.size());
        inDtypes.reserve(stage.inputSlots.size());
        for (auto slot : stage.inputSlots) {
          inPtrs.push_back(slotBufs[slot].get());
          inShapes.push_back(slotShapes[slot]);
          inDtypes.push_back(slotDtypes[slot]);
        }
        std::vector<std::shared_ptr<DeviceBuffer>> outBufs;
        std::vector<std::vector<std::int64_t>> outShapes;
        std::vector<std::string> outDtypes;
        stage.cudnn->executeRaw(inPtrs, inShapes, inDtypes, outBufs, outShapes, outDtypes);
        lastAskEvent.reset();
        for (std::size_t k = 0; k < stage.outputSlots.size(); ++k) {
          slotBufs[stage.outputSlots[k]] = outBufs[k];
          intermediateBufs.push_back(std::move(outBufs[k]));
        }
      }
    }
  }

  if (fast) {
    if (lastAskEvent) {
      rt->retainAsk(std::move(lastAskEvent), tensors, std::move(intermediateBufs));
    } else {
      rt->retainStreamWork(tensors, std::move(intermediateBufs));
    }
  }

  nb::list result;
  for (std::size_t i = 0; i < finalOutputSlots.size(); ++i) {
    const auto &buf = slotBufs[finalOutputSlots[i]];
    nb::object handle;
    for (std::size_t k = 0; k < initialBufs.size(); ++k) {
      if (initialBufs[k].get() == buf.get()) {
        handle = nb::borrow<nb::object>(tensors[k]);
        break;
      }
    }
    if (!handle.is_valid()) {
      for (std::size_t j = 0; j < i; ++j) {
        if (slotBufs[finalOutputSlots[j]].get() == buf.get()) {
          handle = nb::borrow<nb::object>(result[j]);
          break;
        }
      }
    }
    if (!handle.is_valid()) {
      const auto &port = finalOutputPorts[i];
      if (fast) {
        handle = NativeDeviceTensor::makeSubclass(
            tensorType, rt->sessionWeakRef, buf, port.shape, port.dtype,
            port.shapeVec, port.dtypeStr, rt->sessionPtr, rt->descriptorStream);
      } else {
        handle = tensorType(sessionObj, nb::cast(buf), port.shape, port.dtype);
      }
    }
    result.append(std::move(handle));
  }
  return result;
}

}  // namespace fuga::cudnn_rt

void bindCudnnRuntime(nanobind::module_ &module) {
  using namespace fuga::cudnn_rt;

  module.def("cudnn_available", &cudnnAvailable);

  nb::class_<CudnnSessionState>(module, "CudnnSessionState")
      .def(nb::init<std::shared_ptr<fuga::ask_rt::CUDASessionQueue>>(), nb::arg("queue"))
      .def("clear_cache", &CudnnSessionState::clearCache)
      .def("close", &CudnnSessionState::close)
      .def_prop_ro("cache_size", &CudnnSessionState::cacheSize);

  nb::class_<CudnnExecutable>(module, "CudnnExecutable")
      .def("__init__", [](CudnnExecutable *self,
                          std::shared_ptr<CudnnSessionState> state,
                          nb::object sessionObj,
                          nb::object tensorType,
                          std::string entry,
                          nb::sequence outputTypes,
                          bool causal,
                          nb::object scaleObj,
                          std::int64_t maxSq,
                          std::int64_t maxSkv,
                          std::string layout,
                          nb::object artifactDir,
                          std::string opId) {
        std::vector<CudnnExecutable::OutputSpec> specs;
        for (nb::handle ot : outputTypes) {
          CudnnExecutable::OutputSpec s;
          s.shape = nb::cast<std::vector<std::int64_t>>(ot.attr("shape"));
          nb::object dtObj = ot.attr("dtype");
          s.dtype = nb::isinstance<nb::str>(dtObj)
                        ? nb::cast<std::string>(dtObj)
                        : nb::cast<std::string>(dtObj.attr("name"));
          s.shapeTuple = nb::tuple(nb::cast(s.shape));
          s.dtypePyStr = nb::str(s.dtype.c_str());
          specs.push_back(std::move(s));
        }
        bool hasScale = !scaleObj.is_none();
        double scaleVal = hasScale ? nb::cast<double>(scaleObj) : 0.0;
        new (self) CudnnExecutable(std::move(state), std::move(sessionObj), std::move(tensorType),
                                   std::move(entry), std::move(specs), causal, hasScale, scaleVal,
                                   maxSq, maxSkv, std::move(layout), std::move(artifactDir), std::move(opId));
      }, nb::arg("state"), nb::arg("session"), nb::arg("tensor_type"), nb::arg("entry"),
         nb::arg("output_types"), nb::arg("causal"), nb::arg("scale").none(),
         nb::arg("max_s_q"), nb::arg("max_s_kv"), nb::arg("layout"),
         nb::arg("artifact_dir").none(), nb::arg("op_id"))
      .def("run", &CudnnExecutable::run, nb::arg("inputs"))
      .def_ro("id", &CudnnExecutable::opId)
      .def_prop_ro("kind", [](const CudnnExecutable &) { return "external"; })
      .def_ro("source", &CudnnExecutable::sourcePy)
      .def_ro("report", &CudnnExecutable::reportDict)
      .def_ro("artifact_dir", &CudnnExecutable::artifactDir)
      .def_ro("execution_plan", &CudnnExecutable::executionPlanDict);

  nb::class_<CudnnCompositeExecutable>(module, "CudnnCompositeExecutable")
      .def("__init__", [](CudnnCompositeExecutable *self,
                          std::shared_ptr<CudnnSessionState> state,
                          nb::object sessionObj,
                          nb::object tensorType,
                          std::size_t numSlots,
                          std::vector<std::size_t> initialInputSlots,
                          std::vector<std::size_t> finalOutputSlots,
                          nb::sequence initialInputPortDefs,
                          nb::sequence finalOutputPortDefs,
                          std::vector<std::vector<std::int64_t>> slotShapes,
                          std::vector<std::string> slotDtypes,
                          nb::sequence stageDefs,
                          nb::object artifactDir,
                          nb::dict executionPlan,
                          nb::dict report) {
        std::vector<CudnnCompositeExecutable::Stage> stages;
        stages.reserve(nb::len(stageDefs));
        for (nb::handle item : stageDefs) {
          auto d = nb::cast<nb::dict>(item);
          CudnnCompositeExecutable::Stage st;
          st.id = nb::cast<std::string>(d["id"]);
          st.isExternal = (nb::cast<std::string>(d["kind"]) != "owned");
          st.inputSlots = nb::cast<std::vector<std::size_t>>(d["input_slots"]);
          st.outputSlots = nb::cast<std::vector<std::size_t>>(d["output_slots"]);
          if (!st.isExternal) {
            st.program = nb::cast<std::shared_ptr<ask::runtime::PreparedProgram>>(d["program"]);
          } else {
            st.cudnn = nb::cast<std::shared_ptr<CudnnExecutable>>(d["cudnn"]);
          }
          stages.push_back(std::move(st));
        }
        new (self) CudnnCompositeExecutable(
            std::move(state), std::move(sessionObj), std::move(tensorType), numSlots,
            std::move(initialInputSlots), std::move(finalOutputSlots),
            fuga::ask_rt::ProgramPorts::readPorts(initialInputPortDefs),
            fuga::ask_rt::ProgramPorts::readPorts(finalOutputPortDefs),
            std::move(slotShapes), std::move(slotDtypes), std::move(stages),
            std::move(artifactDir), std::move(executionPlan), std::move(report));
      }, nb::arg("state"), nb::arg("session"), nb::arg("tensor_type"), nb::arg("num_slots"),
         nb::arg("initial_input_slots"), nb::arg("final_output_slots"),
         nb::arg("initial_input_ports"), nb::arg("final_output_ports"),
         nb::arg("slot_shapes"), nb::arg("slot_dtypes"), nb::arg("stages"),
         nb::arg("artifact_dir").none(), nb::arg("execution_plan"), nb::arg("report"))
      .def("run", &CudnnCompositeExecutable::run, nb::arg("inputs"))
      .def_ro("source", &CudnnCompositeExecutable::sourcePy)
      .def_ro("report", &CudnnCompositeExecutable::reportDict)
      .def_ro("artifact_dir", &CudnnCompositeExecutable::artifactDir)
      .def_ro("execution_plan", &CudnnCompositeExecutable::executionPlanDict);
}
