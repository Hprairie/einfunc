from torch import einsum
import torch
from src.einfunc.einfunc import einfunc


test_functional_cases = [
    # Non-Uniform test cases
    (
        # Non uniform indexing
        "i i -> i",
        lambda x: x,
        "ii->i",
        ((5, 5),),
        (5,),
    ),
    # The following tests were taken from einops
    # https://github.com/arogozhnikov/einops
    (
        # Basic:
        "b c h w, b w -> b h",
        lambda x, y: x * y,
        "abcd,ad->ac",
        ((2, 3, 4, 5), (2, 5)),
        (2, 4),
    ),
    (
        # Three tensors:
        "b c h w, b w, b c -> b h",
        lambda x, y, z: x * y * z,
        "abcd,ad,ab->ac",
        ((2, 3, 40, 5), (2, 5), (2, 3)),
        (2, 40),
    ),
    (
        # One tensor, and underscores:
        "first_tensor second_tensor -> first_tensor",
        lambda x: x,
        "ab->a",
        ((5, 4),),
        (5,),
    ),
    (
        # Trace (repeated index)
        "i i -> ",
        lambda x: x,
        "aa->",
        ((5, 5),),
        (),
    ),
    (
        # Too many spaces in string:
        " one  two  ,  three four->two  four  ",
        lambda x, y: x * y,
        "ab,cd->bd",
        ((2, 3), (4, 5)),
        (3, 5),
    ),
    # The following tests were inspired by numpy's einsum tests
    # https://github.com/numpy/numpy/blob/v1.23.0/numpy/core/tests/test_einsum.py
    (
        # Trace with other indices
        "i middle i -> middle",
        lambda x: x,
        "aba->b",
        ((5, 10, 5),),
        (10,),
    ),
    (
        # Triple diagonal
        "one one one -> one",
        lambda x: x,
        "aaa->a",
        ((5, 5, 5),),
        (5,),
    ),
    (
        # Axis swap:
        "i j k -> j i k",
        lambda x: x,
        "abc->bac",
        ((1, 2, 3),),
        (2, 1, 3),
    ),
    (
        # Basic summation:
        "index ->",
        lambda x: x,
        "a->",
        ((10,)),
        (()),
    ),
]


def test_shape():
    for test in test_functional_cases:
        equation1, func, equation2, input_shape, output_shape = test
        in_tensors = [torch.rand(x) for x in input_shape]
        einfunc_result = einfunc(*in_tensors, equation1, func)
        einsum_result = einsum(equation2, *in_tensors)

        assert einfunc_result.shape == einsum_result.shape

        assert einfunc_result.shape == output_shape

        # Rounding errors can happen which causes inequality
        assert torch.norm(einfunc_result - einsum_result) < 0.00001


if __name__ == "__main__":
    test_shape()
