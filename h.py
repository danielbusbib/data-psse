"""
Convenience: build h() for IEEE 118 in one call.
"""
import os
import torch

from config import NET_FILE, OBS_LEVEL, GENERATOR_KWARGS
from network import parse_ieee_mat, System, Branch
from h_polar import H_AC


def _default_generator_kwargs(obs=None):
    """Defaults from config.GENERATOR_KWARGS."""
    kw = dict(GENERATOR_KWARGS)
    if obs is not None:
        kw["sample"] = float(obs)
    return kw


def meas_idx_from_generator_kwargs(sys, branch, generator_kwargs=None, obs=None):
    """Build meas_idx for flow, injection, voltage, current.
    With keep_nans=True: full meas_idx, same z shape for all obs levels.
    With keep_nans=False: Bernoulli(sample) sampling, z shape varies with obs.
    """
    if generator_kwargs is None:
        generator_kwargs = _default_generator_kwargs(obs)
    else:
        generator_kwargs = dict(generator_kwargs)
    if obs is not None:
        generator_kwargs["sample"] = float(obs)
    device = None
    keep_nans = bool(generator_kwargs.get("keep_nans", False))
    sample_cfg = generator_kwargs.get("sample", 1.0)

    gen = torch.Generator(device=device)
    gen.manual_seed(666)

    def _p_for_type(mtype: str) -> float:
        if isinstance(sample_cfg, dict):
            p = float(sample_cfg.get(mtype, 1.0))
        else:
            p = float(sample_cfg)
        return max(0.0, min(1.0, p))

    def _bernoulli_mask(n: int, p: float) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0,), device=device, dtype=torch.bool)
        return torch.bernoulli(torch.full((n,), p, device=device), generator=gen).to(torch.bool)

    meas_idx = {}
    nb = len(sys.bus)
    nbr = len(branch.i)
    half = nbr // 2
    bus_mask_all = torch.ones(nb, device=device, dtype=torch.bool)

    if generator_kwargs.get("flow"):
        PQf_mask = torch.cat(
            [torch.ones(half, device=device), torch.zeros(half, device=device)],
            dim=0
        ).to(torch.bool)
        if not keep_nans:
            p = _p_for_type("Pf")
            samp = _bernoulli_mask(int(PQf_mask.sum().item()), p)
            PQf_mask_s = PQf_mask.clone()
            PQf_mask_s[PQf_mask] = samp
            meas_idx["Pf_idx"] = PQf_mask_s
            meas_idx["Qf_idx"] = PQf_mask_s
        else:
            meas_idx["Pf_idx"] = PQf_mask
            meas_idx["Qf_idx"] = PQf_mask

    if generator_kwargs.get("injection"):
        if not keep_nans:
            p = _p_for_type("Pi")
            samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
            inj_mask = bus_mask_all.clone()
            inj_mask[bus_mask_all] = samp
            meas_idx["Pi_idx"] = inj_mask
            meas_idx["Qi_idx"] = inj_mask
        else:
            meas_idx["Pi_idx"] = bus_mask_all
            meas_idx["Qi_idx"] = bus_mask_all

    if generator_kwargs.get("voltage"):
        if not keep_nans:
            p = _p_for_type("Vm")
            samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
            vm_mask = bus_mask_all.clone()
            vm_mask[bus_mask_all] = samp
            meas_idx["Vm_idx"] = vm_mask
        else:
            meas_idx["Vm_idx"] = bus_mask_all

    if generator_kwargs.get("current"):
        cm_mask_all = torch.ones(nbr, device=device, dtype=torch.bool)
        if not keep_nans:
            p = _p_for_type("Cm")
            samp = _bernoulli_mask(int(cm_mask_all.sum().item()), p)
            cm_mask = cm_mask_all.clone()
            cm_mask[cm_mask_all] = samp
            meas_idx["Cm_idx"] = cm_mask
        else:
            meas_idx["Cm_idx"] = cm_mask_all
    # When current=False: do not set Cm_idx (H_AC uses default zeros)

    return meas_idx


def get_h(net_file=NET_FILE, meas_idx=None, generator_kwargs=None, obs=None):
    """Build h(x) for IEEE 118. Returns (h, sys, branch). Uses nets/ieee118_186.mat by default.
    obs: observability level (0–1), e.g. 0.2, 0.4, 0.6, 0.8. Overrides sample when keep_nans=False.
    """
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"IEEE 118 mat file not found: {net_file}")
    data = parse_ieee_mat(net_file)
    sys = System(data['data']['system'])
    branch = Branch(sys.branch)
    if meas_idx is None:
        meas_idx = meas_idx_from_generator_kwargs(sys, branch, generator_kwargs, obs)
    h = H_AC(sys, branch, meas_idx)
    return h, sys, branch
