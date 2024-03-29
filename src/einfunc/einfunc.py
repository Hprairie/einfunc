import types
from typing import Union
from functools import lru_cache

import torch
from functorch.dim import dims


#@lru_cache(maxsize=126)
def _parse_pattern(pattern: str, pass_axis: bool):
    """Parses the pattern of the einstein notation for a tensor operation"""
    if "->" not in pattern:
        raise ValueError("einstein pattern must contain ->")
    input_str, output_str = pattern.split("->")

    # Determine total number of axis and tensor dim map
    inputs = [input.split() for input in input_str.split(",")]
    dim = set.union(*map(set, inputs))
    axis = dims(len(dim))
    if not isinstance(axis, tuple):
        axis = (axis,)
    dim2axis = dict(zip(dim, axis))

    tensor_axis = [[dim2axis[d] for d in exp] for exp in inputs]
    output_axis = [output.split() for output in output_str.split(",")]
    collapse_axis = [list(dim - set(output)) for output in output_axis]
    collapse_axis = [[dim2axis[x] for x in exp] for exp in collapse_axis]
    final_shape = [[dim2axis[x] for x in exp] for exp in output_axis]

    if pass_axis:
        return tensor_axis, collapse_axis, final_shape, axis
    return tensor_axis, collapse_axis, final_shape


def _collapse_function(tensor: torch.Tensor, collapse: str, axis):
    """#Collapses the tensor along the given axis"""
    if len(axis) == 0:
        return tensor
    if collapse == "min":
        return tensor.amin(axis)
    if collapse == "max":
        return tensor.amax(axis)
    if collapse == "sum":
        return tensor.sum(axis)
    if collapse == "prod":
        # pytorch supports reducing only one operation at a time
        for i in axis[::-1]:
            tensor = tensor.prod(i)
        return tensor
    if collapse == "mean":
        return tensor.mean(axis)

    raise NotImplementedError(f"Unknown reduction {collapse} passed to einfunc")


def einfunc(
    *tensors_pattern_function: Union[torch.Tensor, str, types.FunctionType],
    **additional_args,
):
    """einfunc(*tensors, equation, function, additional_args)

        einfunc allows for the readability of einstein notation with the expressivity of your own
    function. einfunc allows the user to pass a function or callable object, along with an einstein
    notation format for indexing in order to allow incredibly flexible and readible tensor
    operations.

    Examples for einfunc operation:

    """
    if len(tensors_pattern_function) <= 2:
        raise ValueError(
            "einfunc takes minimum 3 parameters: the tensor (at least one),\
             the pattern, and the funtion"
        )
    function = tensors_pattern_function[-1]
    pattern = tensors_pattern_function[-2]
    if not isinstance(pattern, str):
        raise ValueError(
            "The second to last argument of einfunc must be a string,\
             representing the einsum pattern"
        )
    if not isinstance(function, types.FunctionType):
        raise ValueError(
            "The last argument of einfunc must be a function or class with __call__ defined"
        )
    collapse = additional_args.get("reduce", "sum")

    assert collapse in [
        "sum",
        "mean",
        "prod",
        "max",
        "min",
    ], f"{collapse} is not one of the allowed reduction types: sum, prod, mean, min, max"

    pass_axis = additional_args.get("indexs", False)

    if not isinstance(pass_axis, bool):
        raise ValueError("indexs must in einfunc must be a boolean type")

    tensors = list(tensors_pattern_function[:-2])

    # Generate the axis
    axis = None
    if pass_axis:
        tensor_axis, collapse_axis, final_shape, axis = _parse_pattern(
            pattern, pass_axis
        )
    else:
        tensor_axis, collapse_axis, final_shape = _parse_pattern(pattern, pass_axis)

    # Apply tensor axis
    for i in range(len(tensors)):
        tensors[i] = tensors[i][tensor_axis[i]]

    # Pass to function (just pass, it should work lol)
    if pass_axis and axis is not None:
        final_tensor = function(*tensors, *axis)
    else:
        final_tensor = function(*tensors)

    if isinstance(final_tensor, tuple):
        final_tensor = list(final_tensor)
    else:
        final_tensor = [final_tensor]

    # apply the mode of collapse
    print(final_tensor, collapse_axis, final_shape)
    ret = [
        (
            _collapse_function(tensor, collapse, caxis).order(*final)
            if len(final)
            else _collapse_function(tensor, collapse, caxis)
        )
        for tensor, caxis, final in zip(final_tensor, collapse_axis, final_shape)
    ]
    print(pattern, final_tensor, collapse_axis, final_shape)
    return ret if len(ret) != 1 else ret[0]
