import math as mt
import torch
from abc import ABC


class LossFunction(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reshape_x(self, x):
        return x

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def update_params(self, *args):
        pass

    def compute_residuals(self, x):
        raise NotImplementedError

    def compute_f(self, x):
        raise NotImplementedError

    def compute_J(self, x):
        raise NotImplementedError

    def compute_grad(self, x):
        raise NotImplementedError


class SELoss(LossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.z = kwargs.get('z')
        self.v = kwargs.get('v')
        self.R = kwargs.get('R')
        self.h_ac = kwargs.get('h_ac')
        self.slk_bus = kwargs.get('slk_bus')
        self.nb = kwargs.get('nb')
        self.norm_H = kwargs.get('norm_H')
        self.lamda = kwargs.get('prior_scale', 1.)

    def update_params(self, *args):
        self.z = args[0]
        self.v = args[1]
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.R = torch.diag(1. / torch.sqrt(self.v))
        self.norm_H = args[5] if len(args) > 5 and args[5] is not None else torch.ones_like(self.z)

    def reshape_x(self, x):
        return self._remove_slack(x)

    def _split_x(self, x):
        T = x[:self.nb]
        V = x[self.nb:]
        return T, V

    def _remove_slack(self, x, dim=0):
        s = int(self.slk_bus[0])
        if dim == 0:
            return torch.cat([x[:s], x[s + 1:]], dim=0)
        else:
            return torch.cat([x[:, :s], x[:, s + 1:]], dim=1)

    def _insert_zero_at_slack(self, vec, dim=0):
        s = int(self.slk_bus[0])
        if dim == 0:
            return torch.cat([vec[:s], torch.zeros(1, device=vec.device, dtype=vec.dtype), vec[s:]], dim=0)
        else:
            return torch.cat([vec[:, :s], torch.zeros(vec.shape[0], 1, device=vec.device, dtype=vec.dtype), vec[:, s + 1:]], dim=1)

    def _build_insert_slack_angle_operator(self, n_full: int):
        s = int(self.slk_bus[0])
        A = torch.zeros((n_full, n_full - 1))
        if s > 0:
            A[0:s, 0:s] = torch.eye(s)
        if s < n_full - 1:
            A[s + 1:, s:] = torch.eye(n_full - s - 1)
        b = torch.zeros((n_full,))
        b[s] = torch.as_tensor(self.slk_bus[1], dtype=torch.get_default_dtype())
        return A, b

    def _insert_slack_angle_linear(self, vec):
        n_full = vec.shape[0] + 1
        A, b = self._build_insert_slack_angle_operator(n_full)
        return A @ vec + b

    def _insert_slack_angle(self, vec):
        s = int(self.slk_bus[0])
        angle = torch.tensor([self.slk_bus[1]], device=vec.device, dtype=torch.get_default_dtype())
        return torch.cat([vec[:s], angle, vec[s:]], dim=0)

    def update_x(self, x, step):
        if step is not None:
            if len(step) < len(x):
                step = self._insert_zero_at_slack(step)
            x = x + step
        return x

    def se_res(self, x):
        z_est = self.h_ac.estimate(x)
        res = (self.R @ (self.z - z_est))
        return res

    def se_jacobian(self, x):
        J = self.h_ac.jacobian(x)
        J = -(self.R @ J)
        J = self._remove_slack(J, dim=1)
        return J

    def se_f(self, x):
        z_est = self.h_ac.estimate(x).detach()
        res_h = self.R @ (self.z - z_est) / self.norm_H
        f_h = .5 * torch.norm(res_h).pow(2)
        return f_h

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step).detach()
        res = self.se_res(x)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        return f_h

    def compute_J(self, x, step=None):
        x = self.update_x(x, step).detach()
        J = self.se_jacobian(x)
        return J

    def compute_grad(self, x, step=None):
        return torch.zeros(self.nb * 2 - 1, device=x.device, dtype=x.dtype)
