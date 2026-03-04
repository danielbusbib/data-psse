"""
IEEE 118 state dataset. Ground truth (T, V) from AC power flow.
"""
import os
import torch
from tqdm import tqdm

from config import NET_FILE, DATA_DIR, NUM_SAMPLES
from network import parse_ieee_mat, System, Branch
from nr_pf import NR_PF


def _init_flat(sys):
    T = torch.deg2rad(torch.full((sys.nb,), 0.0, dtype=torch.get_default_dtype()))
    T[sys.slk_bus[0]] = sys.slk_bus[1]
    V = torch.full((sys.nb,), 1.0, dtype=torch.get_default_dtype())
    return torch.stack([T, V], dim=1)


def _regenerate_PQ(sys, seed=None, load_sigma_frac=0.12, pf_mean=0.95, pf_sigma=0.02, gen_noise_frac=0.08):
    bus = sys.bus
    nb = sys.nb
    Pbase_L = torch.as_tensor(bus['Pl'].fillna(0.0).values, dtype=torch.get_default_dtype())
    if not torch.any(Pbase_L > 0):
        Pbase_L = torch.ones(nb)
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dP = torch.normal(mean=0.0, std=load_sigma_frac, size=(nb,), generator=g)
    P_L = torch.clamp(Pbase_L * (1.0 + dP), min=0.0)
    pf_L = torch.clamp(torch.normal(pf_mean, pf_sigma, (nb,), generator=g), 0.9, 0.99)
    Q_L = P_L * torch.tan(torch.arccos(pf_L))
    Qmin = torch.as_tensor(bus['Qmin'].fillna(-999).values, dtype=torch.get_default_dtype())
    Qmax = torch.as_tensor(bus['Qmax'].fillna(999).values, dtype=torch.get_default_dtype())
    prev_Pg = torch.as_tensor(bus['Pg'].fillna(0.0).values, dtype=torch.get_default_dtype())
    gen_mask = ((torch.isfinite(Qmin) & torch.isfinite(Qmax)) | (prev_Pg > 0))
    gen_idx = torch.nonzero(gen_mask, as_tuple=True)[0]
    Pg = torch.zeros(nb, dtype=torch.get_default_dtype())
    if gen_idx.numel() > 0:
        w = torch.clamp(prev_Pg[gen_idx], min=0.0)
        if w.sum() <= 1e-9:
            w = torch.ones_like(w)
        P_target = P_L.sum()
        Pg_raw = P_target * (w / w.sum())
        Pg_gen = torch.clamp(Pg_raw + Pg_raw * torch.normal(0, gen_noise_frac, Pg_raw.shape, generator=g), min=0.0)
        if Pg_gen.sum() > 1e-9:
            Pg_gen = Pg_gen * (P_target / Pg_gen.sum())
        Pg[gen_idx] = Pg_gen
    Qmin_eff = torch.where(gen_mask, torch.nan_to_num(Qmin, nan=0.0), torch.zeros_like(Qmin))
    Qmax_eff = torch.where(gen_mask, torch.nan_to_num(Qmax, nan=0.0), torch.zeros_like(Qmax))
    Qg = torch.zeros(nb)
    has_range = (Qmax_eff > Qmin_eff)
    mid = 0.5 * (Qmin_eff + Qmax_eff)
    Qg[has_range] = mid[has_range]
    Q_def = Q_L.sum() - Qg.sum()
    if torch.abs(Q_def) > 1e-9 and gen_idx.numel() > 0:
        room_up = torch.clamp(Qmax_eff - Qg, min=0.0)
        room_dn = torch.clamp(Qg - Qmin_eff, min=0.0)
        if Q_def > 0 and room_up.sum() > 1e-12:
            Qg = torch.minimum(Qg + Q_def * (room_up / room_up.sum()), Qmax_eff)
        elif Q_def < 0 and room_dn.sum() > 1e-12:
            Qg = torch.maximum(Qg + Q_def * (room_dn / room_dn.sum()), Qmin_eff)
    Qg[~gen_mask] = 0.0
    return P_L, Q_L, Pg, Qg


def generate_dataset(num_samples=NUM_SAMPLES, net_file=NET_FILE, seed=42, verbose=True):
    """Generate (T, V) states via AC power flow. Saves to data/x.pt."""
    data = parse_ieee_mat(net_file)
    sys = System(data['data']['system'])
    branch = Branch(sys.branch)
    x_init = _init_flat(sys)
    user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}
    g = torch.Generator().manual_seed(seed)
    T_list, V_list = [], []
    for i in tqdm(range(num_samples), desc="Generating states", disable=not verbose):
        curr_sys = sys.copy()
        Pl, Ql, Pg, Qg = _regenerate_PQ(curr_sys, seed=seed + i)
        loads = torch.stack([Pl, Ql], dim=1)
        gens = torch.stack([Pg, Qg], dim=1)
        pf = NR_PF(curr_sys, loads, gens, x_init, user)
        Vc = pf['Vc']
        T_list.append(torch.angle(Vc).to(torch.float32))
        V_list.append(torch.abs(Vc).to(torch.float32))
    T = torch.stack(T_list)
    V = torch.stack(V_list)
    x = torch.cat([T, V], dim=1)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "x.pt")
    torch.save(x, path)
    if verbose:
        print(f"Saved {x.shape} to {path}")
    return x


def load_dataset():
    """Load x = [T, V] from data/x.pt. Regenerates if missing or has fewer than NUM_SAMPLES."""
    path = os.path.join(DATA_DIR, "x.pt")
    if not os.path.exists(path):
        print("Dataset not found. Generating...")
        return generate_dataset()
    x = torch.load(path)
    if x.shape[0] != NUM_SAMPLES:
        print(f"Dataset has {x.shape[0]} samples, config expects {NUM_SAMPLES}. Regenerating...")
        return generate_dataset()
    return x
