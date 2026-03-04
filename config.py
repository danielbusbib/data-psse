"""
All global config for data_play. Single source of truth.
"""
import os

DIM = 100
NUM_SAMPLES = 10_000
Z_DIM = 300  # measurement dimension

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
