# scripts/data_splitter.py
import pandas as pd
import yaml
from pathlib import Path

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

input_file = Path(params['data']['raw_csv'])
output_v1 = Path(params['data']['version_1'])
output_v2 = Path(params['data']['version_2'])
output_full = Path(params['data']['full_data'])
timestamp_col = params['data']['timestamp_col']
split_ratio = params['data_split']['split_ratio']

# Create parent directories
output_v1.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file).sort_values(by=timestamp_col)

# Save the full sorted dataset for poisoning experiments
df.to_csv(output_full, index=False)
print(f"Saved full sorted dataset to {output_full}")

split_point = int(len(df) * split_ratio)
df_v1 = df.iloc[:split_point]
df_v2 = df.iloc[split_point:]

df_v1.to_csv(output_v1, index=False)
df_v2.to_csv(output_v2, index=False)

print(f"Split data into two versions:")
print(f"  - Version 1 ({len(df_v1)} rows) -> {output_v1}")
print(f"  - Version 2 ({len(df_v2)} rows) -> {output_v2}")
