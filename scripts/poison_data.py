# file: scripts/poison_data.py

import pandas as pd
import numpy as np
import os
import argparse

def create_poisoned_datasets(input_file, output_dir, poison_levels, target_col, seed):
    """
    Reads a dataset, flips a percentage of labels, and saves new CSVs.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original dataset
    try:
        df_original = pd.read_csv(input_file)
        print(f"Successfully loaded original dataset from '{input_file}' with {len(df_original)} rows.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    print(f"\nStarting data poisoning with random seed {seed}...")
    
    for level in poison_levels:
        df_poisoned = df_original.copy()
        
        # Determine the number of samples to poison
        num_to_flip = int(len(df_poisoned) * level)
        
        # Randomly select indices to flip
        indices_to_flip = np.random.choice(df_poisoned.index, size=num_to_flip, replace=False)
        
        # Flip the labels (assuming labels are 0 and 1)
        df_poisoned.loc[indices_to_flip, target_col] = 1 - df_poisoned.loc[indices_to_flip, target_col]
        
        # Save the new poisoned dataset
        file_name = f"poisoned_{int(level*100)}pct.csv"
        output_path = os.path.join(output_dir, file_name)
        df_poisoned.to_csv(output_path, index=False)
        
        print(f"  - Created '{output_path}' with {num_to_flip} labels flipped ({level*100:.0f}%).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create poisoned datasets by flipping labels.")
    parser.add_argument("--input-file", type=str, default="data/processed/version_1.csv", help="Path to the original dataset.")
    parser.add_argument("--output-dir", type=str, default="data/poisoned", help="Directory to save the poisoned datasets.")
    parser.add_argument("--target-col", type=str, default="Class", help="Name of the target label column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    poison_levels = [0.02, 0.08, 0.15, 0.30]
    create_poisoned_datasets(args.input_file, args.output_dir, poison_levels, args.target_col, args.seed)

