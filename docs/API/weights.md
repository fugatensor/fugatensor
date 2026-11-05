返回 API 目录：[分类目录](README.md#categories) · 关联模块：[张量构建](tensor.md) · 架构原理：[权重格式与导出](../architecture/weights.md)

# 权重持久化

FugaTensor 提供原生权重容器格式（`.fuga`），支持将任意嵌套结构（dict / list / tuple）与 Tensor 组成的 PyTree 模型参数进行内容寻址序列化。

<a id="save-weights"></a>

## 保存权重

### `ft.save_weights(path, pytree)` / `ft.weights.save(path, pytree)`
将参数字典或 PyTree 保存至单文件容器。

- **特性**：
  - 自动通过 SHA-256 内容寻址实现相同张量数据去重；
  - 记录形状、数据类型与对齐信息；
  - 不保存任何特定的 Python 类、设备状态或算子图。

```python
import fugatensor as ft

params = {
    "layer1": {
        "weight": ft.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": ft.zeros(2),
    },
    "tokens": [ft.ones(4), ft.zeros(4)],
}

ft.save_weights("model.fuga", params)
```

<a id="load-weights"></a>

## 加载权重

### `ft.load_weights(path, *, device=None)` / `ft.weights.load(path, *, device=None)`
恢复之前保存的 PyTree 结构与张量数值。

- **参数**：
  - `path`：权重容器文件路径；
  - `device`：张量载入的目标设备，缺省使用当前默认设备。

```python
loaded = ft.load_weights("model.fuga", device="cpu")
print("Weights:", loaded["layer1"]["weight"].tolist())  # [[1.0, 2.0], [3.0, 4.0]]
print("Bias:", loaded["layer1"]["bias"].tolist())        # [0.0, 0.0]
```

<a id="inspect-weights"></a>

## 清单元数据检查

### `ft.weights.manifest(path)`
无需在内存中物化大张量数据即可高速读取容器的元数据清单：包含 `schema_version`、PyTree 骨架结构与各张量块的 dtype/shape 记录。

```python
meta = ft.weights.manifest("model.fuga")
print("Schema version:", meta["schema_version"])  # 1
```
