返回 API 目录：[分类目录](README.md#categories) · 关联模块：[张量构建](tensor.md) · 架构原理：[多后端执行架构](../architecture/backend.md)

# 设备管理与执行调度

FugaTensor 默认管理设备的全局常驻引擎（Engine）。张量通过延迟图（Lazy Graph）构建，仅在显式求值或消费数值时触发编译与执行。

<a id="devices"></a><a id="default-device"></a>

## 设备对象与默认设备

### `ft.device(name)`
构造设备描述符对象。支持形式：
- `"cpu"`：CPU 设备；
- `"cuda"` 或 `"cuda:N"`：CUDA GPU 设备（缺省序号默认为 0）。

### `ft.set_default_device(device)` / `ft.get_default_device()`
获取或设置当前上下文中后续创建张量的默认设备。

```python
import fugatensor as ft

print("Initial default:", ft.get_default_device())  # cpu

# 切换后续张量默认创建在 CUDA
# ft.set_default_device("cuda")
# x = ft.ones(2, 2)
# print(x.device)  # cuda:0
```

<a id="evaluation"></a>

## 延迟执行与流同步

### `ft.evaluate(*tensors)`
显式物化一个或多个张量背后的待计算图节点。

- **返回**：若传入单个张量返回该张量句柄；若传入多个则返回原张量列表。
- **机制**：收集所有待计算张量的祖先节点，融合成单个静态图，通过后端引擎执行后，将显存/内存物理存储直接就地回填至原张量句柄，保留表达式来源。

```python
import fugatensor as ft

a = ft.tensor([1.0, 2.0])
b = ft.tensor([3.0, 4.0])
c = a + b  # 此时尚未在底层分配结果内存

# 显式触发图编译与运行
ft.evaluate(c)
print(c.tolist())  # [4.0, 6.0]
```

### `ft.synchronize(device=None)`
等待目标设备流上已提交的异步计算任务全部完成，不触发未求值图的物化。`device=None` 时同步当前默认设备。

```python
# 确保 CUDA 设备上在途的 kernel 全部执行完成
ft.synchronize("cpu")
```

<a id="engine-class"></a>

## 独立引擎 `ft.Engine`

用于隔离显存、编译缓存与执行生命周期的高级资源管理类：

```python
import fugatensor as ft

# 创建独立的 CPU 引擎
with ft.Engine(device="cpu") as eng:
    t = eng.tensor([10.0, 20.0])
    print(t.tolist())  # [10.0, 20.0]
    print(eng.cache_info())  # {'compilations': 1, 'cache_hits': 0, ...}
```

- **方法**：
  - `eng.tensor(...)`：在当前引擎创建张量；
  - `eng.evaluate(*tensors)`：使用本引擎的编译缓存和流物化；
  - `eng.synchronize()`：等待本引擎关联的流空闲；
  - `eng.clear_cache()`：清空已编译的 JIT 计划；
  - `eng.close()`：关闭引擎并释放常驻会话与显存池。
