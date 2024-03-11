# Einfunc

[![PyPI - Version](https://img.shields.io/pypi/v/einfunc.svg)](https://pypi.org/project/einfunc)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/einfunc.svg)](https://pypi.org/project/einfunc)

-----

A convient way of applying of functions to tensors. `einfunc` is incredibly similar to `einsum`, but with one key difference. The ability to apply your own custom function instead of multiplication. `einfunc` also allows for the ability to choose how reductions occur within the operation.

`einfunc` is a simple interface, which utilizes pytorch's `torchdim`. I highly recommend checking out `torchdim` as it by far the best way to do tensor operations in pytorch which are incredibly readable. `einfunc` is just a convient way of tapping into `torchdim` with a function similar to `einsum`.

**Table of Contents**

- [Installation](#installation)
- [API](#api)
- [Why you shouldn't use einfunc](#why-not-use)
- [Planned Work](#planned-work)


## Installation

```console
pip install einfunc
```
## API

Using `einfunc` is similar to `einsum` however you also pass a function and a mode of reduction. Take this math equation for example

## Why you shouldn't use einfunc <a name="why-not-use"></a>

## Planned Work <a name="planned-work"></a>

