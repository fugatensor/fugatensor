"""Independent FP64 references for attention values and analytic gradients."""
import math


def reference(q, k, v, *, batch, heads, m, n, d, dv, causal):
    """Small FP64 host oracle over already quantized, flattened values."""
    output = []
    scale = d ** -0.5
    for bh in range(batch * heads):
        for row in range(m):
            keys = min(n, row + 1) if causal else n
            scores = [math.fsum(q[(bh * m + row) * d + dim] *
                                k[(bh * n + col) * d + dim] for dim in range(d)) * scale
                      for col in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            probabilities = [weight / denominator for weight in weights]
            output.extend(math.fsum(probability * v[(bh * n + col) * dv + dim]
                                    for col, probability in enumerate(probabilities))
                          for dim in range(dv))
    return output


def reference_backward(q, k, v, do, *, batch, heads, m, n, d, dv, causal):
    """Analytic softmax-attention gradient for the same quantized values."""
    dq = [0.0] * (batch * heads * m * d)
    dk = [0.0] * (batch * heads * n * d)
    dvalue = [0.0] * (batch * heads * n * dv)
    scale = d ** -0.5
    for bh in range(batch * heads):
        for row in range(m):
            keys = min(n, row + 1) if causal else n
            if keys == 0:
                continue
            scores = [math.fsum(q[(bh * m + row) * d + dim] *
                                k[(bh * n + col) * d + dim] for dim in range(d)) * scale
                      for col in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            probabilities = [weight / denominator for weight in weights]
            output = [math.fsum(probability * v[(bh * n + col) * dv + dim]
                                for col, probability in enumerate(probabilities))
                      for dim in range(dv)]
            delta = math.fsum(do[(bh * m + row) * dv + dim] * output[dim] for dim in range(dv))
            dp = [math.fsum(do[(bh * m + row) * dv + dim] * v[(bh * n + col) * dv + dim]
                            for dim in range(dv)) for col in range(keys)]
            ds = [probabilities[col] * (dp[col] - delta) for col in range(keys)]
            for dim in range(d):
                dq[(bh * m + row) * d + dim] = scale * math.fsum(
                    ds[col] * k[(bh * n + col) * d + dim] for col in range(keys))
            for col in range(keys):
                for dim in range(d):
                    dk[(bh * n + col) * d + dim] += scale * ds[col] * q[(bh * m + row) * d + dim]
                for dim in range(dv):
                    dvalue[(bh * n + col) * dv + dim] += probabilities[col] * do[(bh * m + row) * dv + dim]
    return dq, dk, dvalue


def flatten(value):
    if isinstance(value, (list, tuple)):
        return [element for item in value for element in flatten(item)]
    return [value]


def input_data(shape, *, period, scale):
    """Deterministic nested data; no dependency on Fuga's shape operations."""
    index = 0

    def build(dimensions):
        nonlocal index
        if not dimensions:
            result = ((index * 7) % period - period // 2) * scale
            index += 1
            return result
        return [build(dimensions[1:]) for _ in range(dimensions[0])]

    return build(shape)
