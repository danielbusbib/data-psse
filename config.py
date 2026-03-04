"""
Config for IEEE 118 PSSE data_play.
"""
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(_SCRIPT_DIR, "nets", "ieee118_186.mat")
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
NUM_SAMPLES = 1_000

# Observability level (0–1). With keep_nans=True, obs affects noise variance only; z shape is constant.
OBS_LEVEL = 0.3

# Generator kwargs. keep_nans=True ensures same z shape for all obs levels.
GENERATOR_KWARGS = {
    "flow": True,
    "injection": True,
    "voltage": True,
    "current": False,
    "keep_nans": True,
    "sample": OBS_LEVEL,
    "noise": True,
    "Pf_noise": 0.0004,
    "Qf_noise": 0.0004,
    "Cm_noise": 1e-3,
    "Pi_noise": 16e-3,
    "Qi_noise": 16e-3,
    "Vm_noise": 0.0001,
}
