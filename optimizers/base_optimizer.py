import torch


class SEOptimizer:
    def __init__(self, **kwargs):
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = kwargs.get('max_iter', 100)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb):
        raise NotImplementedError
