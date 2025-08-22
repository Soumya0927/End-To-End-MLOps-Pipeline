import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def main(input_path, output_dir, val_size=0.2, random_seed=42):
    df = pd.read_csv(input_path)
    
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=random_seed, stratify=df["Class"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Train set saved to {train_path}, shape: {train_df.shape}")
    print(f"Val set saved to {val_path}, shape: {val_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed/split")
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.input, args.output_dir, args.val_size)
