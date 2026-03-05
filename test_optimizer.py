"""
Test LM and GN optimizers with flat, random, and perturbed (t) init.
"""
import torch

from config import OBS_LEVEL, GENERATOR_KWARGS, NET_FILE
from dataset import load_dataset
from h import generate_measurements
from network import parse_ieee_mat, System, Branch
from optimizers import LMOpt, GN_se
from utils import init_start_point, perturbed_init_from_true, RMSE


def main():
    # Load data
    x = load_dataset()
    x0_true = x[0]
    T_true, V_true = x0_true[:118], x0_true[118:]

    # Generate measurements
    kwargs = dict(GENERATOR_KWARGS, sample=OBS_LEVEL)
    z, var, _, _, h = generate_measurements(T_true, V_true, **kwargs)

    data = parse_ieee_mat(NET_FILE)
    sys = System(data["data"]["system"])
    slk_bus = sys.slk_bus
    nb = sys.nb

    true_x = torch.cat([T_true, V_true])

    # Init options: flat, random, perturbed with t
    inits = [
        ("flat", torch.cat(init_start_point(sys, how="flat"))),
        ("random", torch.cat(init_start_point(sys, how="random"))),
        ("close (t=1e-4)", perturbed_init_from_true(true_x, t=1e-4)),
        ("far (t=1e-1)", perturbed_init_from_true(true_x, t=1e+0)),
    ]

    gn = GN_se(tol=1e-6, max_iter=200, verbose=False)
    lm = LMOpt(xtol=1e-6, ftol=1e-6, max_iter=200, verbose=False)

    print("=" * 60)
    print("Optimizer test: LM and GN on IEEE 118")
    print(f"obs={OBS_LEVEL}, z.shape={z.shape}")
    print("=" * 60)

    for init_name, x0 in inits:
        x0 = x0.to(z.device)
        slk = slk_bus[0]
        x0[slk] = T_true[slk]
        x0[nb + slk] = V_true[slk]

        x_gn, T_gn, V_gn, conv_gn, it_gn, loss_gn, _ = gn(x0, z, var, slk_bus, h, nb)
        x_lm, T_lm, V_lm, conv_lm, it_lm, loss_lm, _ = lm(x0, z, var, slk_bus, h, nb)

        rmse_gn = RMSE(T_true, V_true, T_gn, V_gn).item()
        rmse_lm = RMSE(T_true, V_true, T_lm, V_lm).item()

        print(f"\nInit: {init_name}")
        print(f"  GN: converged={conv_gn}, it={it_gn}, loss={loss_gn:.4f}, RMSE={rmse_gn:.6f}")
        print(f"  LM: converged={conv_lm}, it={it_lm}, loss={loss_lm:.4f}, RMSE={rmse_lm:.6f}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
