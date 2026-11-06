# Automatic Differentiation and Custom Operators

Back to Architecture Overview: [Functional Automatic Differentiation, `ft.fun`, and Higher-Order Derivatives](README.md#arch-autodiff) · Related topics: [Tensor Model and Tracing Mechanism](tensor_model.md) · [Native Graph Analysis and C++ Boundary](native.md) · [Control Flow and Composable Transformations](../ir/transforms.md)

`ft.grad(outputs, inputs, seed=...)` transforms expressions into new Fuga IR, returning pending Tensors corresponding to the derivatives. Numerical execution is triggered by reading results or calling `ft.evaluate`, and is executed uniformly by ASK.

API: [Automatic Differentiation and Custom Functions](../API/autograd.md)

<a id="functional-autograd"></a>

## 1. Expression Provenance and Pure IR Transformations (`autograd.py`)

A Tensor stores numerical storage alongside expression provenance. After computation results are evaluated, provenance is retained; differentiation can traverse through evaluated intermediate expressions, and intermediate Tensors can also be selected as inputs for differentiation.

1. **Determine Differentiation Boundary**: From the expression dependencies of `outputs`, find the selected `inputs` to form the required Fuga IR. Values with explicit stop-gradients retain numerical dependencies while truncating the corresponding differentiation path.
2. **Check Gradient Seed**: When the seed is omitted for 0-D outputs, a scalar `1.0` of the same device and dtype is used; non-scalars must provide a seed with matching shape, dtype, and Engine. Seeds for multiple outputs use a tuple/list of the same length; container types are interchangeable.
3. **Transform IR**: Construct a [`vjp`](../ir/transforms.md#jvp-and-vjp) module based on C++ liveness analysis. Cotangents on shared paths are merged via `fuga.add`, while inputs without contribution are padded with zeros.
4. **Return Expressions**: Derivatives, like results of normal operators, are pending Tensors that can continue participating in operations and differentiation. Each invocation retains its own result; the user explicitly decides whether to sum multiple derivatives.
5. **Derivative Plan Cache and Fast Path**: Derivative plans are cached by expression module and port index. Standard operator graphs directly reuse derivative templates upon a cache hit, verifying function definition versions only when containing custom `ft.fun` calls.

Graphs preceding and succeeding standard Python scalar conditions are connected via Tensor provenance, and derivatives propagate along the path actually taken in the current execution. `detach()` and `stop_gradient()` clearly demarcate differentiation boundaries, while numerical dependencies continue participating in execution.

`grad` operates on runtime Tensors. `detach` and `stop_gradient` support both runtime Tensors and TraceTensors, and can appear inside `ft.fun.forward` or explicit VJPs. Runtime Tensors truncate provenance; isolated tracing generates `fuga.stop_gradient`, which preserves values, dtypes, and shapes while halting JVP/VJP propagation. When lowered to ASK, input values are mapped directly, as differentiation boundaries are handled by Fuga IR transformations.

<a id="custom-fun"></a>

## 2. `ft.fun` Forward and Pure VJP Rules (`function.py`)

Custom functions express the forward pass using regular methods, and can optionally provide a VJP that receives primals, outputs, and cotangents:

```python
import fugatensor as ft

class Square(ft.fun):
    def forward(self, x):
        return ft.square(x)

    def vjp(self, primals, outputs, cotangents):
        (x,) = primals
        (dy,) = cotangents
        return (2 * x * dy,)
```

- When `op(x)` is invoked, `forward` is isolated-traced, saving the submodule, `vjp_rule` / `vjp_name`, and `definition_key` into a `CallSpec`, and inserting `fuga.call` in the outer graph. Modifying instance configuration or replacing methods after graph construction causes old graphs to raise an error.
- Autograd configuration resides in immutable instance attributes, while runtime Tensors are passed as arguments. After modifying mutable state in closures or global variables, call `op.clear_cache()` and reconstruct the graph; such captured content is not covered by definition checks.
- During differentiation, an explicit VJP constructs new SSA expressions from the three tuples `primals`, `outputs`, and `cotangents`. Primals contain all Tensor inputs, outputs contain all public results; cotangents for unused floating-point outputs are zero Tensors, and cotangents for integer and boolean outputs are `None`. Only gradients for floating-point Tensor inputs are returned, with quantity, shape, and dtype checked elementwise. When an explicit VJP is absent, derivatives are generated via the primitive rules of the forward subgraph.
- VJP can invoke [`ft.triton.kernel`](backend.md#cuda-and-triton). Unused inputs are pruned during compilation, and ASK schedules external kernels into the execution plan; numerical computation is still performed by these kernels.

<a id="higher-order"></a>

## 3. Closure of Derivative Graphs and Higher-Order Differentiation

Derivative graphs generated by existing JVP/VJP rules still utilize the same set of Fuga IR primitives. For example, multiplication VJP generates two `fuga.mul` operations, cotangents of shared inputs are accumulated with `fuga.add`; floating-point cast and reshape generate inverse cast and reshape, respectively. The applicability of each rule is documented in the [Primitive Rules Table](../ir/transforms.md#primitive-registry).

The expression returned by `first = ft.grad(y, x)` can be passed further to `ft.grad(first, x)`. Evaluation of derivatives does not delete provenance. Higher-order differentiation depends on primitive rules along the actual active paths; undefined derivatives of opaque external kernels are rejected. When an explicit VJP is composed of closed primitives, one can continue differentiating the derivative expression even if the forward pass contains external kernels; only opaque nodes on the active path of the derivative graph restrict higher-order differentiation. Standard forward passes support automatic JVP; invocations with explicit VJPs currently reject active JVPs to preserve the semantics of custom derivatives.

Stop-gradient operates similarly on derivative graphs. For example, when a VJP returns `stop_gradient(x) * dy`, subsequent differentiation with respect to `x` yields zero. If all paths between an external kernel and the target derivative pass through stop-gradient nodes, the differentiation transform no longer requires rules for that kernel, while numerical computation is preserved according to dependencies.

<a id="preflight-validation"></a>

## 4. Preflight Validation during Transformation

`ft.grad` inspects requests and required rules before returning derivative expressions:

1. Validate the structure, dtype, shape, and Engine of outputs, inputs, and seeds;
2. Locate active differentiation paths based on C++ [`analyze_function`](native.md#native-analyze);
3. Construct and validate VJP IR; active bare `fuga.external` lacking rules raises an immediate error, and return values of custom VJPs must match the contract for floating-point inputs.

Once transformation succeeds, callers can compose derivative expressions and evaluate them uniformly. Invalid requests are rejected prior to numerical execution.
