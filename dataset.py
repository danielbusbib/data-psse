"""
Simple dataset: 10k x's of dimension DIM.
No external deps beyond torch.
"""
import os
import torch

try:
    from data_play.config import DIM, NUM_SAMPLES, DATA_DIR
except ImportError:
    from config import DIM, NUM_SAMPLES, DATA_DIR


def _sample_from_cov(eigvals: torch.Tensor, n_samples: int, g: torch.Generator) -> torch.Tensor:
    """Sample (n_samples, len(eigvals)) from N(0, Q @ diag(eigvals) @ Q.T) with random Q."""
    nb = eigvals.numel()
    Q, _ = torch.linalg.qr(torch.randn(nb, nb, generator=g))
    L = Q * eigvals.sqrt().unsqueeze(0)  # (nb, nb)
    z = torch.randn(n_samples, nb, generator=g, dtype=torch.float32)
    return z @ L.T


def generate_x(num_samples: int = NUM_SAMPLES, dim: int = DIM, seed: int = 42) -> torch.Tensor:
    """Generate x = [T, V] with covariance eigenvalues spanning high and very small.
    T (angles): N(0, Sigma_T), eigvals_T decay 1 -> 1e-6.
    V (voltages): 1 + N(0, Sigma_V), eigvals_V decay 0.01 -> 1e-6, clamped to [0.9, 1.1].
    """
    g = torch.Generator().manual_seed(seed)
    nb = dim // 2

    # T: eigenvalues from e^1.5 down to 1e-6 (wide spread)
    eig_T = torch.exp(torch.linspace(1.5, -6 * torch.log(torch.tensor(10.)).item(), nb))
    eig_T[0] = 0.0  # slack will be fixed
    T = _sample_from_cov(eig_T, num_samples, g)
    T[:, 0] = 0.0  # slack angle

    # V: eigenvalues from ~0.01 down to 1e-6 (small scale, wide spread)
    eig_V = 0.01 * torch.exp(torch.linspace(0, -6 * torch.log(torch.tensor(10.)).item(), nb))
    eig_V[0] = 0.0  # slack fixed
    V_delta = _sample_from_cov(eig_V, num_samples, g)
    V = torch.clamp(1.0 + V_delta, 0.9, 1.1)
    V[:, 0] = 1.0  # slack voltage

    x = torch.cat([T, V], dim=1)
    return x


def save_dataset(num_samples: int = NUM_SAMPLES, dim: int = DIM, seed: int = 42):
    """Generate and save x to data/x.pt."""
    os.makedirs(DATA_DIR, exist_ok=True)
    x = generate_x(num_samples, dim, seed)
    path = os.path.join(DATA_DIR, "x.pt")
    torch.save(x, path)
    print(f"Saved {x.shape} to {path}")
    return x


def load_dataset() -> torch.Tensor:
    """Load x from data/x.pt. Run save_dataset() first if missing."""
    path = os.path.join(DATA_DIR, "x.pt")
    if not os.path.exists(path):
        print("Dataset not found. Generating...")
        return save_dataset()
    return torch.load(path)


def print_T_V_cov_eigenvalues(x: torch.Tensor):
    """Split x=[T,V], compute covariances, print eigenvalue stats."""
    nb = x.shape[1] // 2
    T = x[:, :nb]
    V = x[:, nb:]

    def _cov_eigstats(name: str, data: torch.Tensor):
        data_c = data - data.mean(dim=0)
        cov = (data_c.T @ data_c) / (data.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).float()
        eigvals = eigvals[torch.isfinite(eigvals)]
        eigvals = eigvals[eigvals > 1e-12]  # drop near-zero from slack fix
        if eigvals.numel() == 0:
            print(f"  {name}: no finite positive eigenvalues")
            return
        print(f"  {name} cov eigenvalues: max={eigvals.max():.6e}, min={eigvals.min():.6e}, "
              f"mean={eigvals.mean():.6e}, std={eigvals.std():.6e}, count={eigvals.numel()}")

    print("True T, V covariance eigenvalues:")
    _cov_eigstats("T", T)
    _cov_eigstats("V", V)


if __name__ == "__main__":
    x = save_dataset()
    print(f"x shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
    print_T_V_cov_eigenvalues(x)
