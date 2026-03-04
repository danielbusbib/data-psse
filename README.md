# data_play

Minimal dataset + differentiable forward model for experiments.

- **Dataset**: 10k x's, dim 118 (`data/x.pt`)
- **Forward model**: `z = h(x) + noise`, h differentiable

## Quick start

```python
from data_play.dataset import load_dataset, save_dataset
from data_play.forward_model import ForwardModel

# Generate/save dataset (run once)
save_dataset()  # -> data/x.pt

# Load
x = load_dataset()  # (10000, 118)

# Forward model
model = ForwardModel()
z = model.observe(x[:32], noise_std=0.01)  # (32, 200)
```

## Files

- `config.py` – all globals (DIM, NUM_SAMPLES, Z_DIM, DATA_DIR)
- `dataset.py` – generate/load x
- `forward_model.py` – h(x) + noise, torch-differentiable
