"""
Test: load dataset, generate measurements with obs-dependent noise.
"""
import torch

from config import OBS_LEVEL, GENERATOR_KWARGS
from dataset import load_dataset
from h import generate_measurements


def main():
    x = load_dataset()
    x0 = x[0]
    T0, V0 = x0[:118], x0[118:]

    kwargs = dict(GENERATOR_KWARGS, sample=OBS_LEVEL)
    z, var, _, _, h = generate_measurements(T0, V0, **kwargs)

    print(f"x: {x.shape}")
    print(f"z: {z.shape}  (obs={OBS_LEVEL})")
    print(f"J: {h.jacobian(x0).shape}")

    x0_grad = x0.clone().requires_grad_(True)
    h.estimate(x0_grad).sum().backward()
    print(f"grad ok: {x0_grad.grad is not None}")


if __name__ == "__main__":
    main()
