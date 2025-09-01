"""
STAC ingestion utilities to fetch Sentinel-2 / Landsat-8/9 / Sentinel-1 chips
to local .npz files.

Usage:

  python -m utils.stac_ingest \
    --out_dir data/npz \
    --lat 37.6 --lon -122.3 \
    --size 128 --start 2020-01-01 --end 2020-12-31

"""

from typing import List, Tuple
import argparse
import os
from pathlib import Path
import numpy as np

from pystac_client import Client
import rasterio



EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


def _bbox_from_center(lat: float, lon: float, meters: float = 1280.0) -> List[float]:
    # Approx meter per deg
    dlat = meters / 111_000.0
    dlon = meters / (111_000.0 * max(0.1, np.cos(np.deg2rad(lat))))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def _read_cog_window(url: str, size: int) -> np.ndarray:
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(url) as ds:
            # center crop to square of given size in pixels by resampling
            data = ds.read(out_shape=(ds.count, size, size), resampling=rasterio.enums.Resampling.bilinear)
            data = np.transpose(data, (1, 2, 0)).astype(np.float32)
            return data


def fetch_chips(out_dir: str,
                lat: float,
                lon: float,
                size: int,
                start: str,
                end: str) -> None:
    """Fetch Sentinel-2 chips from STAC and save to .npz files."""
    os.makedirs(out_dir, exist_ok=True)
    bbox = _bbox_from_center(lat, lon, meters=size * 10.0)  # 10m scale approx.
    client = Client.open(EARTH_SEARCH)

    result = {}
    # Sentinel-2 only
    search = client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}", limit=50)
    items = list(search.get_items())
    frames = []
    tss = []
    for it in items:
        assets = it.assets
        band_list = ["B02", "B03", "B04", "B08", "B11"]
        chips = []
        for b in band_list:
            if b not in assets:
                chips = []
                break
            chips.append(_read_cog_window(assets[b].href, size))
        if not chips:
            continue
        arr = np.stack(chips, axis=-1)
        frames.append(arr)
        tss.append(np.datetime64(it.datetime).astype("datetime64[ms]").astype(np.int64))
    if frames:
        s2 = np.stack(frames, axis=0)
        result["sentinel2"] = s2
        result["ts_sentinel2"] = np.array(tss, dtype=np.float64)

    name = f"chip_lat{lat:.3f}_lon{lon:.3f}_sz{size}.npz"
    np.savez_compressed(Path(out_dir) / name, **result)
    print(f"Saved {name} with keys: {list(result.keys())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    args = p.parse_args()
    fetch_chips(args.out_dir, args.lat, args.lon, args.size, args.start, args.end)


if __name__ == "__main__":
    main()
