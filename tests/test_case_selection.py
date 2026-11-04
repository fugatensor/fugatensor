"""Check the generated coverage promise independently of numerical execution."""
from itertools import combinations, product

from tests.support.matmul import HOST_AXES, host_cases


def test_host_matmul_representatives_cover_all_factor_pairs():
    selected = host_cases()
    exhaustive = host_cases(exhaustive=True)
    assert len(selected) == len(set(selected)) == 26
    assert len(exhaustive) == 1536
    for i, j in combinations(range(len(HOST_AXES)), 2):
        assert {(row[i], row[j]) for row in selected} == set(product(HOST_AXES[i], HOST_AXES[j]))
    assert set(selected) <= set(exhaustive)


def test_attention_representatives_cover_pairs_and_causal_tail_composition():
    from tests.support.attention import attention_cases
    axes = (('fp16', 'bf16'), ('tail', 'wide', 'batched'), ('false', 'true'))
    selected = attention_cases(*axes, ('plain', 'scaled'))
    exhaustive = attention_cases(*axes, ('plain', 'scaled'), exhaustive=True)
    assert len(selected) == len(set(selected)) == 8
    assert len(exhaustive) == 24
    assert set(selected) <= set(exhaustive)
    plain = [row for row in selected if row[-1] == 'plain']
    for i, j in combinations(range(len(axes)), 2):
        assert {(row[i], row[j]) for row in plain} == set(product(axes[i], axes[j]))
    assert {(row[0], row[1], row[2]) for row in selected if row[-1] == 'scaled'} == {
        (dtype, 'tail', 'true') for dtype in axes[0]}
