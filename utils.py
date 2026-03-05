"""
Utilities for data_play: init, RMSE.
"""
import torch


def init_start_point(sys, data=None, how='flat',
                    flat_init=(0, 1), random_init=(0.3, 1, 1e-2)):
    """Return (T, V) for x0. how: 'flat' | 'random'."""
    if how == 'flat':
        T = torch.deg2rad(torch.full((sys.nb,), flat_init[0], dtype=torch.get_default_dtype()))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = torch.full((sys.nb,), flat_init[1], dtype=torch.get_default_dtype())

    elif how == 'random':
        theta = torch.pi * random_init[0]
        T = torch.empty(sys.nb).uniform_(-theta, theta)
        T[sys.slk_bus[0]] = sys.slk_bus[1]

        mu, sigma = random_init[1], torch.sqrt(torch.tensor(random_init[2]))
        V = torch.normal(mu, sigma, size=(sys.nb,))
        V[sys.slk_bus[0]] = 1.0

    else:
        T = torch.deg2rad(torch.full((sys.nb,), flat_init[0], dtype=torch.get_default_dtype()))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = torch.full((sys.nb,), flat_init[1], dtype=torch.get_default_dtype())

    return T, V


def perturbed_init_from_true(true_x: torch.Tensor, t: float, generator=None) -> torch.Tensor:
    """
    x0 = true_x + t * (u + iv) / ||u + iv||_inf
    where u, v ~ N(0, I_n). For real state, we use Re(term) = u / ||u+iv||_inf.
    """
    n = true_x.numel()
    g = generator
    u = torch.randn(n, dtype=true_x.dtype, device=true_x.device, generator=g)
    v = torch.randn(n, dtype=true_x.dtype, device=true_x.device, generator=g)
    norm_inf = torch.max(torch.sqrt(u ** 2 + v ** 2)) + 1e-12
    term = u / norm_inf
    return true_x + t * term


def RMSE(T_true: torch.Tensor, V_true: torch.Tensor, T_est: torch.Tensor, V_est: torch.Tensor):
    """Relative RMSE: ||u_est - u_true|| / ||u_true||."""
    u_true = V_true.to(dtype=torch.get_default_dtype()) * torch.exp(1j * T_true.to(torch.get_default_dtype()))
    u_est = V_est.to(dtype=torch.get_default_dtype()) * torch.exp(1j * T_est.to(torch.get_default_dtype()))
    num = torch.linalg.norm(u_est - u_true)
    den = torch.linalg.norm(u_true) + 1e-12
    return (num / den).real
