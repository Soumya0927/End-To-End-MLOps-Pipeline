# scripts/data_poisoner.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

input_file = Path(params['data']['poison_source'])
output_dir = Path(params['data']['poisoned_datasets_dir'])
target_col = params['data']['target_col']
poison_levels = params['poisoning']['levels']
seed = params['project']['random_seed']

# Reproducibility
np.random.seed(seed)

output_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(input_file)

for level in poison_levels:
    df_poisoned = df.copy()
    n_samples_to_flip = int(level * len(df_poisoned))
    
    indices_to_flip = np.random.choice(df_poisoned.index, n_samples_to_flip, replace=False)
    
    df_poisoned.loc[indices_to_flip, target_col] = 1 - df_poisoned.loc[indices_to_flip, target_col]
    
    output_path = output_dir / f"poisoned_{int(level*100)}pct.csv"
    df_poisoned.to_csv(output_path, index=False)
    print(f"Created poisoned dataset at {level*100:.0f}% level: {output_path}")
