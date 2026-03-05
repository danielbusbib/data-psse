import torch
from tqdm import tqdm
from optimizers.base_optimizer import SEOptimizer


class GN_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(GN_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-7)
        self.max_iter = kwargs.get('max_iter', 500)
        self.verbose = kwargs.get('verbose', True)
        self.output_cond = kwargs.get('output_cond', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb, norm_H=None):
        device = z.device

        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()

        x = torch.cat([T, V])
        all_x = [x]
        R = torch.diag(1.0 / v)
        norm_H = norm_H if norm_H is not None else torch.ones_like(z)
        converged = False
        it, xtol, loss = 0, torch.inf, torch.inf
        conds_gain = []
        pbar = tqdm(range(self.max_iter), desc=f'Optimizing with GN', leave=True, colour='green')
        for it in pbar:
            z_est = h_ac.estimate(T, V)
            J = h_ac.jacobian(T, V)
            delta_z = (z - z_est) / norm_H

            J = torch.cat([J[:, :slk_bus[0]], J[:, slk_bus[0] + 1:]], dim=1)

            JT_R = J.T @ R
            lhs = JT_R @ J
            rhs = JT_R @ delta_z

            try:
                delta_x_reduced = torch.linalg.solve(lhs, rhs)
                if self.output_cond:
                    conds_gain.append(torch.linalg.cond(lhs).item())
            except torch.linalg.LinAlgError:
                break
            delta_x = torch.cat([
                delta_x_reduced[:slk_bus[0]],
                torch.tensor([0.0], device=device, dtype=delta_x_reduced.dtype),
                delta_x_reduced[slk_bus[0]:]
            ])

            xtol = torch.norm(delta_x, p=torch.inf).item()
            error = delta_z.reshape(-1, 1)
            loss = (error.T @ R @ error).item()
            x = x + delta_x
            T = x[:nb]
            V = x[nb:]
            all_x.append(x.clone().detach())
            if xtol <= self.tol:
                converged = True
                break

            pbar.set_postfix(xtol=f"{xtol:.4e}", loss=f"{loss:.4f}")

        return x, T, V, converged, it, loss, all_x
