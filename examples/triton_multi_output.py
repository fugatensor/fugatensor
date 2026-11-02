"""普通标量和双输出 Triton kernel；接入契约见 docs/architecture/04-ASK桥接与多后端执行.md。"""
from pathlib import Path
import fugatensor as ft


def main():
    operation = ft.triton.kernel(Path(__file__).with_name('kernels.py'),
        entry='affine_pair', input='inp', outputs={'out': {}, 'square': {}},
        scalars=lambda spec: {'N': ('i32', spec.numel), 'scale': ('fp32', 2.5)},
        grid=lambda spec: ((spec.numel + 127) // 128,))
    with ft.Engine(device='cuda'):
        x = ft.tensor([1., 2., 3.])
        y, square = operation(x)
        results = ft.evaluate(y + 1, square)
        values = [result.tolist() for result in results]
        assert values == [[5., 7.5, 10.], [1., 4., 9.]]
        print(values)


if __name__ == '__main__':
    main()
