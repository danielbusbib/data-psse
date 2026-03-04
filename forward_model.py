"""
Differentiable forward model: z = h(x) + noise.
h(x) is a simple nonlinear map (torch-differentiable).
"""
import os
import torch
import torch.nn as nn

try:
    from data_play.config import DIM, Z_DIM, DATA_DIR
except ImportError:
    from config import DIM, Z_DIM, DATA_DIR


class ForwardModel(nn.Module):
    """
    h(x) = W2 @ tanh(W1 @ x + b1) + b2
    Fully differentiable. Returns z = h(x) + noise.
    """

    def __init__(self, x_dim: int = DIM, z_dim: int = Z_DIM, seed: int = 42):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(64, x_dim, generator=g) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(64))
        self.W2 = nn.Parameter(torch.randn(z_dim, 64, generator=g) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(z_dim))

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward: h(x)."""
        out = torch.tanh(x @ self.W1.T + self.b1)
        return out @ self.W2.T + self.b2

    def observe(self, x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """z = h(x) + noise."""
        z = self.h(x)
        if noise_std > 0:
            z = z + noise_std * torch.randn_like(z, device=z.device, dtype=z.dtype)
        return z


def load_or_create_model(x_dim: int = DIM, z_dim: int = Z_DIM, seed: int = 42):
    """Load saved model or create new one."""
    path = os.path.join(DATA_DIR, "forward_model.pt")
    if os.path.exists(path):
        m = ForwardModel(x_dim, z_dim, seed)
        m.load_state_dict(torch.load(path))
        return m
    return ForwardModel(x_dim, z_dim, seed)


def save_model(model: ForwardModel):
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(DATA_DIR, "forward_model.pt"))


if __name__ == "__main__":
    from dataset import load_dataset

    x = load_dataset()
    x = x[:5]  # batch of 5

    model = ForwardModel()
    z = model.observe(x, noise_std=0.01)
    print(f"x: {x.shape} -> z: {z.shape}")
    print(f"h(x) requires_grad: {model.h(x).requires_grad}")
