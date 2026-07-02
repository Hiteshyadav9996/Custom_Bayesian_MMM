"""Small standalone Watchtower-style MMM teaching package."""

from toy_mmm.config import ToyConfig, default_config
from toy_mmm.hdf5_io import load_hdf5_bundle, write_hdf5_bundle
from toy_mmm.normalization import normalize_raw_data
from toy_mmm.simulate import simulate_raw_data
from toy_mmm.train import train_model

__all__ = [
    "ToyConfig",
    "default_config",
    "load_hdf5_bundle",
    "normalize_raw_data",
    "simulate_raw_data",
    "train_model",
    "write_hdf5_bundle",
]
