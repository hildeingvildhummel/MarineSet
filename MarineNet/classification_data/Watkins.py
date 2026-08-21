import os
import shutil
import pandas as pd
from tqdm import tqdm

SOURCE_DIR = "Data/watkins"
OUTPUT_DIR = "Data/watkins_split"

splits = {
    "train": "annotations.train.csv",
    "val": "annotations.valid.csv",
    "test": "annotations.test.csv",
}

for split, csv_file in splits.items():
    df = pd.read_csv(os.path.join(SOURCE_DIR, csv_file))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        src = row["path"]
        label = row["label"]

        # Create class directory
        dst_dir = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy audio
        dst = os.path.join(dst_dir, os.path.basename(src))

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

print("Done!")