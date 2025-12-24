"""Preprocessing utilities for OlmoEarth pretrain dataset."""

import argparse
from pathlib import Path
import pandas as pd
import tarfile
import numpy as np
from tqdm import tqdm


def inspect_dataset(data_dir: str, csv_path: str = None):
    data_dir = Path(data_dir)
    
    print(f"Inspecting dataset in: {data_dir}")
    print("=" * 60)
    
    if csv_path is None:
        parent_dir = data_dir.parent
        csv_candidates = list(parent_dir.glob("*.csv"))
        if csv_candidates:
            csv_path = csv_candidates[0]
        else:
            print("Warning: No CSV file found")
            csv_path = None
    
    if csv_path:
        print(f"\nCSV metadata: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    
    tar_files = sorted(data_dir.glob("*.tar"))
    print(f"\nTar files: {len(tar_files)}")
    for tar_file in tar_files[:5]:
        print(f"  {tar_file.name}")
    if len(tar_files) > 5:
        print(f"  ... and {len(tar_files) - 5} more")
    
    if tar_files:
        print(f"\nInspecting first tar file: {tar_files[0].name}")
        with tarfile.open(tar_files[0], 'r') as tar:
            members = tar.getmembers()
            print(f"  Total members: {len(members)}")
            
            npy_members = [m for m in members if m.name.endswith('.npy')]
            print(f"  .npy files: {len(npy_members)}")
            
            if npy_members:
                first_member = npy_members[0]
                print(f"\n  First .npy file: {first_member.name}")
                file_obj = tar.extractfile(first_member)
                if file_obj:
                    data = np.load(file_obj)
                    print(f"    Shape: {data.shape}")
                    print(f"    Dtype: {data.dtype}")
                    print(f"    Min: {np.nanmin(data):.4f}")
                    print(f"    Max: {np.nanmax(data):.4f}")
                    print(f"    Mean: {np.nanmean(data):.4f}")
                    print(f"    Has NaN: {np.isnan(data).any()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect OlmoEarth dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/olmoearth_pretrain_dataset/10_landsat_monthly",
        help="Directory containing tar files",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV metadata file",
    )
    
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.csv_path)


if __name__ == "__main__":
    main()

