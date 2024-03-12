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
        None,
        ((5, 5),),
        (5,),
    ),
    # The following tests were taken from einops
    # https://github.com/arogozhnikov/einops
    (
        # Basic:
        "b c h w, b w -> b h",
        lambda x, y: x * y,
        "abcd,ad->abcd",
        (1, 3),
        ((2, 3, 4, 5), (2, 5)),
        (2, 4),
    ),
    (
        # Three tensors:
        "b c h w, b w, b c -> b h",
        lambda x, y, z: x * y * z,
        "abcd,ad,ab->abcd",
        (1, 3),
        ((2, 3, 40, 5), (2, 5), (2, 3)),
        (2, 40),
    ),
    (
        # One tensor, and underscores:
        "first_tensor second_tensor -> first_tensor",
        lambda x: x,
        "ab->ab",
        (1,),
        ((5, 4),),
        (5,),
    ),
    (
        # Trace (repeated index)
        "i i -> ",
        lambda x: x,
        "aa->a",
        (0,),
        ((5, 5),),
        (),
    ),
    (
        # Too many spaces in string:
        " one  two  ,  three four->two  four  ",
        lambda x, y: x * y,
        "ab,cd->abcd",
        (0, 2),
        ((2, 3), (4, 5)),
        (3, 5),
    ),
    # The following tests were inspired by numpy's einsum tests
    # https://github.com/numpy/numpy/blob/v1.23.0/numpy/core/tests/test_einsum.py
    (
        # Trace with other indices
        "i middle i -> middle",
        lambda x: x,
        "aba->ab",
        (0,),
        ((5, 10, 5),),
        (10,),
    ),
    (
        # Triple diagonal
        "one one one -> one",
        lambda x: x,
        "aaa->a",
        None,
        ((5, 5, 5),),
        (5,),
    ),
    (
        # Axis swap:
        "i j k -> j i k",
        lambda x: x,
        "abc->bac",
        None,
        ((1, 2, 3),),
        (2, 1, 3),
    ),
    (
        # Basic summation:
        "index ->",
        lambda x: x,
        "a->a",
        (0,),
        ((10,)),
        (()),
    ),
]

REDUCTION_TYPES = ["sum", "prod", "max", "min", "mean"]


def test_shape():
    for test in test_functional_cases:
        equation1, func, equation2, dimension_reduction, input_shape, output_shape = (
            test
        )
        for reduction in REDUCTION_TYPES:

            in_tensors = [torch.rand(x) for x in input_shape]
            einfunc_result = einfunc(*in_tensors, equation1, func, reduce=reduction)
            einsum_result = einsum(equation2, *in_tensors)

            if dimension_reduction is not None:
                if reduction == "prod":
                    # pytorch supports reducing only one operation at a time
                    for i in list(sorted(dimension_reduction))[::-1]:
                        einsum_result = einsum_result.prod(i)
                elif reduction == "sum":
                    einsum_result = einsum_result.sum(dimension_reduction)
                elif reduction == "max":
                    einsum_result = einsum_result.amax(dimension_reduction)
                elif reduction == "min":
                    einsum_result = einsum_result.amin(dimension_reduction)
                elif reduction == "mean":
                    einsum_result = einsum_result.mean(dimension_reduction)

            assert einfunc_result.shape == einsum_result.shape

            assert einfunc_result.shape == output_shape

            # Rounding errors can happen which causes inequality
            assert torch.norm(einfunc_result - einsum_result) < 0.00001


if __name__ == "__main__":
    test_shape()
