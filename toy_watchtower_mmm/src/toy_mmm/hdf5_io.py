from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from toy_mmm.normalization import NormalizedToyData


def write_hdf5_bundle(data: NormalizedToyData, output_dir: Path | str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / "toy_mmm_data.h5"
    metadata_path = output_dir / "metadata.json"
    with h5py.File(hdf5_path, "w") as h5:
        h5.create_dataset("features", data=data.features)
        h5.create_dataset("price", data=data.price)
        h5.create_dataset("vehicle_spends", data=data.vehicle_spends)
        h5.create_dataset("target", data=data.target)
        h5.create_dataset("investment_axis_scale", data=data.investment_axis_scale)
        h5.create_dataset("region_idx", data=data.region_idx)
        h5.create_dataset("brand_idx", data=data.brand_idx)
        h5.create_dataset("channel_idx", data=data.channel_idx)
        h5.create_dataset("train_week_mask", data=data.train_week_mask.astype(np.bool_))
        h5.create_dataset("val_week_mask", data=data.val_week_mask.astype(np.bool_))
    metadata_path.write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")
    return hdf5_path, metadata_path


def load_hdf5_bundle(hdf5_path: Path | str, metadata_path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hdf5_path = Path(hdf5_path)
    metadata_path = Path(metadata_path)
    with h5py.File(hdf5_path, "r") as h5:
        tensors = {key: h5[key][()] for key in h5.keys()}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return tensors, metadata
