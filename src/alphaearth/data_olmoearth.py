import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
import rasterio
from rasterio.io import MemoryFile


class OlmoEarthDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        csv_path: Optional[str] = None,
        patch_size: int = 256,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        num_bands: int = 7,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.num_bands = num_bands
        self.tar_cache = {}
        
        if csv_path is None:
            parent_dir = self.data_dir.parent
            csv_candidates = list(parent_dir.glob("*.csv"))
            if csv_candidates:
                csv_path = str(csv_candidates[0])
            else:
                same_dir_candidates = list(self.data_dir.glob("*.csv"))
                if same_dir_candidates:
                    csv_path = str(same_dir_candidates[0])
                else:
                    raise FileNotFoundError(f"No CSV file found in {parent_dir} or {self.data_dir}")
        
        print(f"Loading metadata from {csv_path}...")
        self.metadata = pd.read_csv(csv_path)
        print(f"Loaded {len(self.metadata)} samples")
        
        tar_files = sorted(self.data_dir.glob("*.tar"))
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in {data_dir}")
        
        self.tar_files = tar_files
        
        self.samples = []
        print("Indexing tar files...")
        for tar_idx, tar_path in enumerate(tqdm(self.tar_files, desc="Indexing")):
            with tarfile.open(tar_path, 'r') as tar:
                members = [m for m in tar.getmembers() 
                          if m.isfile() and (m.name.endswith('.npy') or 
                                             m.name.endswith('.npz') or
                                             m.name.endswith('.tif') or
                                             m.name.endswith('.tiff') or
                                             'data' in m.name.lower())]
                for member in members:
                    self.samples.append((tar_idx, member.name))
        
        print(f"Found {len(self.samples)} samples across {len(self.tar_files)} tar files")
        
        if len(self.samples) != len(self.metadata):
            print(f"Warning: {len(self.samples)} samples in tar files but {len(self.metadata)} in CSV")
            min_len = min(len(self.samples), len(self.metadata))
            if min_len < len(self.samples):
                self.samples = self.samples[:min_len]
            if min_len < len(self.metadata):
                self.metadata = self.metadata.iloc[:min_len]
    
    def _get_tar_file(self, tar_idx: int) -> tarfile.TarFile:
        if tar_idx not in self.tar_cache:
            tar_path = self.tar_files[tar_idx]
            self.tar_cache[tar_idx] = tarfile.open(tar_path, 'r')
        return self.tar_cache[tar_idx]
    
    def _load_from_tar(self, tar_idx: int, member_name: str) -> np.ndarray:
        tar = self._get_tar_file(tar_idx)
        try:
            member = tar.getmember(member_name)
        except KeyError:
            all_members = tar.getmembers()
            matching = [m for m in all_members if member_name in m.name or m.name.endswith(member_name.split('/')[-1])]
            if matching:
                member = matching[0]
            else:
                raise ValueError(f"Could not find {member_name} in tar file")
        
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Could not extract {member_name} from tar")
        
        if member_name.endswith(('.tif', '.tiff')):
            file_bytes = file_obj.read()
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    data = dataset.read()
                    data = np.transpose(data, (1, 2, 0))
            return data
        elif member_name.endswith('.npz'):
            npz_data = np.load(file_obj)
            keys = list(npz_data.keys())
            if keys:
                data = npz_data[keys[0]]
            else:
                raise ValueError(f"Empty npz file: {member_name}")
        else:
            data = np.load(file_obj)
        
        return data
    
    def _normalize_landsat(self, data: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return data
        
        if data.ndim == 4:
            normalized = np.zeros_like(data, dtype=np.float32)
            for c in range(data.shape[-1]):
                band = data[..., c]
                band_min = np.nanmin(band)
                band_max = np.nanmax(band)
                if band_max > band_min:
                    normalized[..., c] = (band - band_min) / (band_max - band_min)
                else:
                    normalized[..., c] = band
            return normalized
        else:
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            return data
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        try:
            if isinstance(timestamp_str, (int, float)):
                if timestamp_str > 1e12:
                    return float(timestamp_str)
                else:
                    return float(timestamp_str) * 1000.0
            
            dt = pd.to_datetime(timestamp_str)
            return dt.timestamp() * 1000.0
        except Exception as e:
            print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
            return 1577836800000.0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tar_idx, member_name = self.samples[idx]
        data = self._load_from_tar(tar_idx, member_name)
        
        if idx < len(self.metadata):
            row = self.metadata.iloc[idx]
        else:
            row = self.metadata.iloc[idx % len(self.metadata)]
        
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]
        
        if data.ndim != 4:
            raise ValueError(f"Unexpected data shape: {data.shape}, expected (T, H, W, C)")
        
        T, H, W, C = data.shape
        
        if C > self.num_bands:
            data = data[..., :self.num_bands]
        elif C < self.num_bands:
            padding = np.zeros((T, H, W, self.num_bands - C), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=-1)
        
        if H != self.patch_size or W != self.patch_size:
            zoom_factors = (1.0, self.patch_size / H, self.patch_size / W, 1.0)
            data = zoom(data, zoom_factors, order=1)
        
        data = self._normalize_landsat(data)
        data_tensor = torch.from_numpy(data).float()
        
        timestamps = []
        timestamp_cols = ['timestamp', 'time', 'date', 'datetime', 'start_time']
        base_timestamp = None
        
        for col in timestamp_cols:
            if col in row:
                base_timestamp = self._parse_timestamp(row[col])
                break
        
        if base_timestamp is None:
            base_timestamp = 1577836800000.0
        
        month_ms = 30 * 24 * 3600 * 1000.0
        for t in range(T):
            timestamps.append(base_timestamp + t * month_ms)
        
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        if T > 0:
            valid_start = timestamps[0]
            valid_end = timestamps[-1] + month_ms
        else:
            valid_start = base_timestamp
            valid_end = base_timestamp + (365 * 24 * 3600 * 1000.0)
        
        source_name = "landsat"
        
        return {
            "source_data": {source_name: data_tensor},
            "timestamps": {source_name: timestamps_tensor},
            "valid_period": (valid_start, valid_end),
        }
    
    def __del__(self):
        if hasattr(self, 'tar_cache'):
            for tar in self.tar_cache.values():
                tar.close()


def create_olmoearth_dataloader(
    data_dir: str,
    csv_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = 256,
    normalize: bool = True,
    shuffle: bool = True,
    num_bands: int = 7,
) -> DataLoader:
    dataset = OlmoEarthDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        patch_size=patch_size,
        normalize=normalize,
        num_bands=num_bands,
    )
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        first_sample = batch[0]
        source_name = next(iter(first_sample["source_data"].keys()))
        
        source_tensors = [sample["source_data"][source_name] for sample in batch]
        max_time = max(t.shape[0] for t in source_tensors)
        
        padded_tensors = []
        for tensor in source_tensors:
            if tensor.shape[0] < max_time:
                T, H, W, C = tensor.shape
                padding = torch.zeros(max_time - T, H, W, C, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=0)
            padded_tensors.append(tensor)
        
        collated_sources = {source_name: torch.stack(padded_tensors)}
        
        timestamps_list = [sample["timestamps"][source_name] for sample in batch]
        max_time_ts = max(len(t) for t in timestamps_list)
        
        padded_timestamps = []
        for ts in timestamps_list:
            if len(ts) < max_time_ts:
                last_ts = ts[-1] if len(ts) > 0 else torch.tensor(0.0)
                padding = torch.full((max_time_ts - len(ts),), float(last_ts), dtype=ts.dtype)
                ts = torch.cat([ts, padding])
            padded_timestamps.append(ts)
        
        collated_timestamps = {source_name: torch.stack(padded_timestamps)}
        
        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "valid_periods": [sample["valid_period"] for sample in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

