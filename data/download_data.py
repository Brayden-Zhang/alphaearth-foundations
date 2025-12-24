"""
Download specific files from the OlmoEarth pretraining dataset.

This script downloads some tar files for Landsat monthly data. :


Data Download Location:
    Files will be downloaded to: ./data/olmoearth_pretrain_dataset/
    You can customize this by modifying the local_dir parameter.

Usage:
    Login first: huggingface-cli login
    Then run: python data/download_data.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "allenai/olmoearth_pretrain_dataset"
repo_type = "dataset"

output_dir = Path(__file__).parent / "olmoearth_pretrain_dataset"
output_dir.mkdir(exist_ok=True)

files_to_download = [
    "10_landsat_monthly/0000.tar",
    "10_landsat_monthly/0001.tar",
    "10_landsat_monthly/0002.tar",
    "10_landsat_monthly/0003.tar",
    "10_landsat_monthly/0004.tar",
    "10_landsat_monthly/0005.tar",
    "10_landsat_monthly/0006.tar",
    "10_landsat_monthly/0007.tar",
    "10_landsat_monthly/0008.tar",
    "10_landsat_monthly/0009.tar",
    "10_landsat_monthly.csv",
]

print(f"Downloading specific files from {repo_id}")
print(f"Output directory: {output_dir.absolute()}\n")

for file_path in files_to_download:
    print(f"Downloading {file_path}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type=repo_type,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Downloaded to: {downloaded_path}")
    except Exception as e:
        print(f"  ✗ Error downloading {file_path}: {e}")

print(f"\nDownload complete! Files are in: {output_dir.absolute()}")

