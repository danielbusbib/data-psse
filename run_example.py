"""
Test: load dataset, build h(), compute z = h(x) + noise.
"""
import torch

from config import OBS_LEVEL
from dataset import load_dataset
from h import get_h


def main():
    # Load or generate dataset
    x = load_dataset()
    print(f"x: {x.shape}  (T: {x.shape[1]//2}, V: {x.shape[1]//2})")

    # Build h() for IEEE 118. With keep_nans=True, z shape is constant (726) for all obs.
    h, sys, branch = get_h(obs=OBS_LEVEL)

    # Test h(x)
    x0 = x[0]
    z = h.estimate(x0)
    print(f"z: {z.shape}")

    # With noise
    noise_std = 0.01
    z_noisy = z + noise_std * torch.randn_like(z)
    print(f"z_noisy: {z_noisy.shape}")

    # Jacobian
    J = h.jacobian(x0)
    print(f"J: {J.shape}")

    # Grad check
    x0_grad = x0.clone().requires_grad_(True)
    z_grad = h.estimate(x0_grad)
    z_grad.sum().backward()
    print(f"grad ok: {x0_grad.grad is not None}")


if __name__ == "__main__":
    main()
