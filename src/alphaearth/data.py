"""
Data loaders
- Only Sentinel-2 inputs (B2, B3, B4, B8, B11) at 10 m
- NPZ dataset option (from STAC chips)
- Simple collate that pads time to max within batch
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any




class AEFDataset(Dataset):
    
    def __init__(self, num_samples: int = 1000, patch_size: int = 128, num_frames: int = 16):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.num_frames = num_frames
        # Only Sentinel-2 for simplicity
        self.input_sources = ["sentinel2"]
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a single training sample following paper specifications."""
        
        # Support period (input data range) - up to 1 year as per paper
        support_start_ms = 1577836800000.0  # 2020-01-01 in ms
        support_end_ms = support_start_ms + (365 * 24 * 3600 * 1000)  # 1 year later
        
        # Valid period (summary period) - can be different from support
        valid_start_ms = support_start_ms + np.random.uniform(0, 6 * 30 * 24 * 3600 * 1000)  # Random start within 6 months
        valid_end_ms = valid_start_ms + (365 * 24 * 3600 * 1000)  # 1 year summary
        
        # Generate timestamps (S2 only)
        timestamps = {"sentinel2": self._generate_timestamps(support_start_ms, support_end_ms, self.num_frames)}
        
        # Generate input source data (S2 only)
        input_data = {}
        for source in self.input_sources:
            num_frames = len(timestamps[source])
            input_data[source] = self._generate_source_data(source, num_frames, is_input=True)
        source_data = input_data
        
        return {
            "source_data": source_data,
            "timestamps": timestamps,  # Only input source timestamps
            "valid_period": (valid_start_ms, valid_end_ms),
        }
    
    def _generate_timestamps(self, start_ms: float, end_ms: float, num_frames: int) -> torch.Tensor:
        """Generate random timestamps within period."""
        timestamps = np.random.uniform(start_ms, end_ms, num_frames)
        timestamps = np.sort(timestamps)
        return torch.tensor(timestamps, dtype=torch.float32)

    def _generate_source_data(self, source: str, num_frames: int, is_input: bool = True, num_channels_from_datasource: int = 5) -> torch.Tensor:
        """Generate synthetic data for a source."""
        return torch.rand(num_frames, self.patch_size, self.patch_size, num_channels_from_datasource)


def create_aef_dataloader(num_samples: int = 1000, batch_size: int = 4, num_workers: int = 2, num_frames: int = 16, patch_size: int = 128) -> DataLoader:
    """Create AlphaEarth Foundations dataloader with proper collation."""
    
    dataset = AEFDataset(num_samples=num_samples, patch_size=patch_size, num_frames=num_frames)
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function handling variable-length sequences."""
        
        # Collate S2
        collated_sources = {}
        source = "sentinel2"
        source_tensors = []
        for sample in batch:
            source_tensors.append(sample["source_data"][source])
        max_time = max(t.shape[0] for t in source_tensors)
        padded_tensors = []
        for tensor in source_tensors:
            if tensor.shape[0] < max_time:
                padding = torch.zeros(max_time - tensor.shape[0], *tensor.shape[1:])
                tensor = torch.cat([tensor, padding], dim=0)
            padded_tensors.append(tensor)
        collated_sources[source] = torch.stack(padded_tensors)
        
        # Collate timestamps for S2
        collated_timestamps = {}
        timestamps_list = []
        for sample in batch:
            timestamps_list.append(sample["timestamps"]["sentinel2"])
        max_time_ts = max(len(t) for t in timestamps_list)
        padded_timestamps = []
        for ts in timestamps_list:
            if len(ts) < max_time_ts:
                last_ts = ts[-1] if len(ts) > 0 else torch.tensor(0.0)
                padding = torch.full((max_time_ts - len(ts),), float(last_ts))
                ts = torch.cat([ts, padding])
            padded_timestamps.append(ts)
        collated_timestamps["sentinel2"] = torch.stack(padded_timestamps)
        
        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "valid_periods": [sample["valid_period"] for sample in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


class AEFNPZDataset(Dataset):
    """
    Dataset that reads pre-extracted chips saved as .npz files.
    Each NPZ should contain:
      - sentinel2: (T, H, W, 5)
      - ts_sentinel2: (T,) in ms
    """
    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.glob('*.npz')])
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = np.load(path)
        # Build dicts
        src = {}
        ts = {}
        s = "sentinel2"
        if s in data:
            arr = data[s].astype(np.float32)
            src[s] = torch.from_numpy(arr)
            ts_key = f"ts_{s}"
            if ts_key in data:
                ts[s] = torch.from_numpy(data[ts_key].astype(np.float32))
        # Use a default 1-year valid/support centered at median ts
        if ts:
            med = float(np.median(np.concatenate([t.numpy() for t in ts.values()])))
        else:
            med = 1577836800000.0
        vp = (med - 15552000000.0, med + 15552000000.0)  # +/- 180 days
        sp = (vp[0], vp[1])
        return {
            "source_data": src,
            "timestamps": ts,
            "valid_period": vp,
        }


def create_aef_dataloader_from_npz(root: str, batch_size: int = 2, num_workers: int = 4) -> DataLoader:
    dataset = AEFNPZDataset(root)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated_sources = {}
        collated_timestamps = {}
        s = "sentinel2"
        tensors = [sample["source_data"][s] for sample in batch]
        ts_list = [sample["timestamps"][s] for sample in batch]
        max_t = max(t.shape[0] for t in tensors)
        padded = []
        padded_ts = []
        for x, t in zip(tensors, ts_list):
            if x.shape[0] < max_t:
                pad_x = torch.zeros(max_t - x.shape[0], *x.shape[1:])
                x = torch.cat([x, pad_x], dim=0)
                last = t[-1] if t.numel() else torch.tensor(0.0)
                pad_t = last.repeat(max_t - t.shape[0])
                t = torch.cat([t, pad_t], dim=0)
            padded.append(x)
            padded_ts.append(t)
        collated_sources[s] = torch.stack(padded)
        collated_timestamps[s] = torch.stack(padded_ts)

        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "valid_periods": [sample["valid_period"] for sample in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
