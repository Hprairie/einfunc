# Einfunc

[![PyPI - Version](https://img.shields.io/pypi/v/einfunc.svg)](https://pypi.org/project/einfunc)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/einfunc.svg)](https://pypi.org/project/einfunc)

-----

A convenient way of applying functions to tensors. `einfunc` is incredibly similar to `einsum`, but with one key difference. The ability to apply your own custom function instead of multiplication. `einfunc` also allows for the ability to choose how reductions occur within the operation.

`einfunc` is a simple interface, which utilizes Pytorch's [torchdim](https://github.com/facebookresearch/torchdim). I highly recommend checking out torchdim as it is by far the best way to do readable tensor operations in pytorch. `einfunc` is just a convenient way of tapping into torchdim with a function similar to `einsum`.

**Table of Contents**

- [Installation](#installation)
- [API](#api)
- [Why you shouldn't use einfunc](#why-not-use)
- [Why you should use einfunc](#why-yes-use)
- [Additional Examples](#additional-examples)
- [Planned Work](#planned-work)
- [Acknowledgements](#acknowledgements)


## Installation

`einfunc` requires torch >= 2.0 and python >= 3.11.

```console
pip install einfunc
```
## API

Using `einfunc` is similar to `einsum` however you also pass a function and a mode of reduction. Take this math equation for example.

$$ \frac{1}{K} \sum_j^K \left(\log (z_j) / \prod_i^B x_i^2 - e^{y_{i,j}} \right)$$

We can use einfunc to represent this math equation with 2 lines of code.
```python
inner_exp = einfunc(x, y, 'b, b k -> k', lambda a, b : a ** 2 - torch.exp(b), reduce='prod')
final_exp = einfunc(z, inner_exp, 'k, k -> ', lambda a, b : torch.log(a) / b, reduce='mean')
```
### Creating functions
While lambda functions are simple, any function can work as long as it takes the correct number of inputs. For example when looking at the following expression:

```python
inner_exp = einfunc(x, y, 'i, i j -> j', lambda a, b : a ** 2 - torch.exp(b), reduce='prod')
```
`x` maps to `a` and `y` maps to `b`. This means that the order of the function variables is passed in the order that tensors are passed to einfunc.

### Reducing
Currently, einfunc supports 5 different types of reduction. 
- Mean

$$ \frac{1}{I} \sum_i^I x_{i,j} - y_{k, i} $$

```python
result = einfunc(x, y, 'i j, k i -> j k', lambda a, b : a - b, reduce='mean')
```
  
- Sum
  
$$ \sum_i^I x_{i,j} - y_{k, i} $$

```python
result = einfunc(x, y, 'i j, k i -> j k', lambda a, b : a - b, reduce='sum')
```
  
- Prod
  
$$ \prod_i^I x_{i,j} - y_{k, i} $$

```python
result = einfunc(x, y, 'i j, k i -> j k', lambda a, b : a - b, reduce='prod')
```

- Max

$$ Max(x_{i,j} - y_{k, i}, \quad dim=i) $$

```python
result = einfunc(x, y, 'i j, k i -> j k', lambda a, b : a - b, reduce='max')
```
- Min

$$ Min(x_{i,j} - y_{k, i}, \quad dim=i) $$

```python
result = einfunc(x, y, 'i j, k i -> j k', lambda a, b : a - b, reduce='min')
```

One thing to note is that if `reduce` is not passed then 'sum' is assumed by einfunc.

## Why you shouldn't use einfunc <a name="why-not-use"></a>

Einfunc is just a convenient way of interfacing with PyTorch and Torchdim. This creates some overhead when operating on tensors, compared to vanilla operations and torchdim operations. This means that it will be much faster to use vanilla pytorch operations if doing a simple operation, or just use torchdim if trying to do something more complex.

## Why you should use einfunc <a name="why-yes-use"></a>

It's convenient and slightly more readable than torchdim IMO. Understanding exactly what is happening in an operation can be hard, and einfunc makes it a lot simpler by boiling operations down to a single expression, while using einstein notation to indicate indexing.

## Additional Examples <a name="additional-examples"></a>

Coming Soon :)

## Planned Work <a name="planned-work"></a>

Currently, einfunc does support parenthesis and ellipses. I will be working on implementing this as soon as I can.

## Acknowledgements

Check out [einops](https://github.com/arogozhnikov/einops) and [torchdim](https://github.com/facebookresearch/torchdim).
