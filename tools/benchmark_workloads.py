"""Shared Fuga/PyTorch workloads, generated cases and independent references."""
from dataclasses import dataclass
from itertools import product


WORKLOADS = ("elementwise", "reduction_chain", "fork_join", "matmul")
SIZES = ("small", "large")
DTYPES = ("fp32", "fp16")
LAYOUTS = ("row_major", "transpose")


@dataclass(frozen=True)
class Case:
    workload: str
    size: str
    dtype: str
    layout: str
    rows: int
    columns: int
    depth: int = 0
    branches: int = 1

    @property
    def shape(self):
        return self.rows, self.columns


def cases(workloads=WORKLOADS, sizes=SIZES, dtypes=("fp32",),
          layouts=("row_major",), depths=(8,), branches=(4,)):
    """Cross independent axes; depth and branch count apply to their workload."""
    tensor_shapes = {"small": (32, 64), "large": (512, 256)}
    matrix_sides = {"small": 16, "large": 128}
    for workload, size, dtype, layout in product(workloads, sizes, dtypes, layouts):
        rows, columns = tensor_shapes[size]
        if workload == "matmul":
            rows = columns = matrix_sides[size]
        for depth, width in product(depths if workload == "reduction_chain" else (0,),
                                    branches if workload == "fork_join" else (1,)):
            yield Case(workload, size, dtype, layout, rows, columns, depth, width)


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def input_data(case):
    """Contiguous physical inputs; transpose is an operation inside the graph."""
    rows, columns = case.shape
    if case.workload == "matmul":
        # Rank-one operands with varied rows/columns and alternating inner signs.
        # Each product is a multiple of 1/16; K <= 128 keeps fp16 sums exact.
        left = [[((row % 5) - 2) / 4 * (1 if inner % 2 == 0 else -1)
                 for inner in range(columns)] for row in range(rows)]
        right = [[(1 if inner % 2 == 0 else -1) * ((column % 7) - 3) / 4
                  for column in range(columns)] for inner in range(columns)]
        inputs = (left, right)
    else:
        divisor = 4 if case.workload == "elementwise" else 1
        inputs = tuple([[(row * 3 + column * 5 + branch * 2) % 7 / divisor - 3 / divisor
                         for column in range(columns)] for row in range(rows)]
                       for branch in range(case.branches))
    return tuple(_transpose(value) for value in inputs) if case.layout == "transpose" else inputs


def make_function(case, backend, module):
    """Return a callable that constructs a fresh result on every invocation."""
    orient = (lambda value: value.T) if case.layout == "transpose" else (lambda value: value)
    if backend == "fuga":
        maximum = lambda value: module.amax(value, axis=1, keepdims=True)
    else:
        maximum = lambda value: module.amax(value, dim=1, keepdim=True)
    broadcast = module.broadcast_to
    shape = case.shape

    if case.workload == "elementwise":
        def operation(value):
            value = orient(value)
            return (value + 0.5) * (value - 0.25) + value
    elif case.workload == "reduction_chain":
        def operation(value):
            base = orient(value)
            current = base
            for _ in range(case.depth):
                current = broadcast(maximum(current), shape) + base
            return current
    elif case.workload == "fork_join":
        def operation(*values):
            results = []
            for value in values:
                value = orient(value)
                results.append(broadcast(maximum(value), shape) + value)
            result = results[0]
            for value in results[1:]:
                result = result + value
            return result
    elif case.workload == "matmul":
        def operation(left, right):
            return orient(left) @ orient(right)
    else:
        raise ValueError(f"unknown workload: {case.workload}")
    return operation


def expected(case):
    """Independent scalar references; all default fp16/fp32 steps are exact."""
    inputs = input_data(case)
    if case.layout == "transpose":
        inputs = tuple(_transpose(value) for value in inputs)
    rows, columns = case.shape
    if case.workload == "elementwise":
        result = [(value + 0.5) * (value - 0.25) + value for row in inputs[0] for value in row]
    elif case.workload == "reduction_chain":
        # Each stage adds the fixed row maximum to the previous row maximum.
        result = [case.depth * max(row) + value for row in inputs[0] for value in row]
    elif case.workload == "fork_join":
        maxima = [[max(row) for row in value] for value in inputs]
        result = [sum(maxima[branch][row] + value[row][column]
                      for branch, value in enumerate(inputs))
                  for row in range(rows) for column in range(columns)]
    elif case.workload == "matmul":
        left, right = inputs
        # The alternating signs cancel for every term of the dot product.
        result = [columns * left[row][0] * right[0][column]
                  for row in range(rows) for column in range(columns)]
    else:
        raise ValueError(f"unknown workload: {case.workload}")
    return case.shape, result
