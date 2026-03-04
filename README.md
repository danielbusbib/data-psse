# data-psse

IEEE 118 power system state estimation playground. Single dataset + single h() function.

- **Dataset**: Ground truth states (T, V) from AC power flow, saved as `data/x.pt`
- **h(x)**: Torch-differentiable measurement model for IEEE 118

## Quick start

```python
from dataset import load_dataset, generate_dataset
from h_polar import H_AC
from network import parse_ieee_mat, System, Branch

# Generate dataset (run once, ~2 min for 10k samples)
generate_dataset(num_samples=10_000)

# Load
x = load_dataset()  # (N, 236) = [T, V] for 118 buses

# Build h()
from h import get_h
h, sys, branch = get_h(obs=0.2)  # obs: 0–1, e.g. 0.2, 0.4, 0.6, 0.8; set in config.OBS_LEVEL

# z = h(x) + noise (z dim: 160 for obs=0.2, 726 for full)
z = h.estimate(x[0]) + 0.01 * torch.randn_like(h.estimate(x[0]))
```

## Files

- `config.py` – paths, NUM_SAMPLES, OBS_LEVEL (0–1) for testing different observability levels
- `dataset.py` – generate/load states via AC power flow
- `h_polar.py` – h(x) measurement model (torch-differentiable)
- `h.py` – get_h() builds h(x) with configurable meas_idx
- `network.py` – parse IEEE mat, System, Branch
- `nr_pf.py` – Newton-Raphson power flow
- `compose.py` – measurement composition (Pi, Qi, Pf, Qf, Cm, Vm)
- `nets/ieee118_186.mat` – IEEE 118 case

## Dependencies

```
torch
scipy
pandas
tqdm
```
