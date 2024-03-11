import torch
from src.einfunc import einfunc


def test_function():
    def test1(tensor):
        pass

    tests = [test1]

    for test in tests:
        tensor = torch.rand(10, 20, 30, 40)
