# scripts/download_data.py
import gdown
import yaml
import pandas as pd
from pathlib import Path

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

gdrive_url = params['data']['gdrive_url']
output_file = Path(params['data']['raw_csv'])

# Create parent directory if it doesn't exist
output_file.parent.mkdir(parents=True, exist_ok=True)

# Download the file
print(f"Downloading from {gdrive_url} to {output_file}...")
gdown.download(id=gdrive_url.split('/')[-2], output=str(output_file), quiet=False)
print("Download complete.")

# Quick check
df = pd.read_csv(output_file)
print(f"Dataset loaded with shape: {df.shape}")
