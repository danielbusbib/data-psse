"""Quick sanity check: load data, run forward model."""
from dataset import load_dataset, save_dataset, print_T_V_cov_eigenvalues
from forward_model import ForwardModel

if __name__ == "__main__":
    save_dataset()
    x = load_dataset()
    print_T_V_cov_eigenvalues(x)    
    print(f"x: {x.shape}")

    model = ForwardModel()
    z = model.observe(x[:32], noise_std=0.01)
    print(f"z: {z.shape}")

    # Grad check
    x0 = x[:1].clone().requires_grad_(True)
    z0 = model.h(x0)
    z0.sum().backward()
    print(f"grad ok: {x0.grad is not None}")
