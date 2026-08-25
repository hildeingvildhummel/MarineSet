import pandas as pd
from pathlib import Path
import shutil


DATA_DIR = Path("/scratch-shared/hhummel/DCLDE2027_windows_5s")


# ============================================================
# Load metadata
# ============================================================

df = pd.read_csv(
    DATA_DIR / "windows.csv"
)

# ============================================================
# Extract recording datetime from original_filepath
# ============================================================

df["recording_datetime"] = pd.to_datetime(
    df["original_filepath"].str.extract(
        r"\.(\d{8}_\d{6}Z)\.flac$"
    )[0],
    format="%Y%m%d_%H%M%SZ",
    errors="coerce"
)

# Check for timestamps that could not be extracted
missing_dates = df["recording_datetime"].isna().sum()

print(
    f"Could not extract recording date from "
    f"{missing_dates:,} windows"
)

df = df.dropna(
    subset=["recording_datetime"]
)

df["recording_datetime"] = pd.to_datetime(
    df["recording_datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=["recording_datetime"]
)

# ============================================================
# Get recordings in chronological order
# ============================================================

recordings = (
    df.groupby("recording")["recording_datetime"]
    .min()
    .sort_values()
    .reset_index()
)


# Number of windows per recording
recording_counts = (
    df.groupby("recording")
    .size()
    .reset_index(name="n_windows")
)

recordings = recordings.merge(
    recording_counts,
    on="recording"
)


# ============================================================
# Find chronological boundary at ~80% of windows
# ============================================================

recordings["cumulative_windows"] = (
    recordings["n_windows"].cumsum()
)

target = len(df) * 0.8

split_idx = (
    recordings["cumulative_windows"] >= target
).idxmax()

split_time = recordings.loc[
    split_idx,
    "recording_datetime"
]

train_recordings = set(
    recordings.loc[
        :split_idx,
        "recording"
    ]
)


# ============================================================
# Split
# ============================================================

train_df = df[
    df["recording"].isin(train_recordings)
].copy()

test_df = df[
    ~df["recording"].isin(train_recordings)
].copy()


# ============================================================
# Print split information
# ============================================================

print("=" * 60)
print("TEMPORAL SPLIT")
print("=" * 60)

print(
    f"Total: {len(df):,}"
)

print(
    f"Train: {len(train_df):,} "
    f"({len(train_df) / len(df):.1%})"
)

print(
    f"Test:  {len(test_df):,} "
    f"({len(test_df) / len(df):.1%})"
)

print(
    "\nTemporal boundary:",
    split_time
)


# ============================================================
# Save split metadata
# ============================================================

train_df.to_csv(
    DATA_DIR / "train.csv",
    index=False
)

test_df.to_csv(
    DATA_DIR / "test.csv",
    index=False
)


# ============================================================
# Organize 5-second WAV files
# ============================================================

def organize_windows(split_df, split_name):

    split_dir = DATA_DIR / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print(f"ORGANIZING {split_name.upper()}")
    print("=" * 60)

    copied = 0
    missing = 0

    for _, row in split_df.iterrows():

        # ----------------------------------------------------
        # Source file
        # ----------------------------------------------------

        source = DATA_DIR / row["file"]

        # ----------------------------------------------------
        # Determine class folder
        # ----------------------------------------------------

        if str(row["multilabel"]).upper() == "TRUE":
            label = "multilabel"
        else:
            label = str(row["primary_label"])

        # ----------------------------------------------------
        # Destination
        # ----------------------------------------------------

        destination_dir = (
            split_dir / label
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            destination_dir / source.name
        )

        # ----------------------------------------------------
        # Check source
        # ----------------------------------------------------

        if not source.exists():

            print(
                f"WARNING: File not found: {source}"
            )

            missing += 1
            continue

        # ----------------------------------------------------
        # Copy file
        # ----------------------------------------------------

        shutil.copy2(
            source,
            destination
        )

        copied += 1

    print(
        f"Copied:  {copied:,}"
    )

    print(
        f"Missing: {missing:,}"
    )


# ============================================================
# Organize train and test
# ============================================================

organize_windows(
    train_df,
    "train"
)

organize_windows(
    test_df,
    "test"
)


print()
print("=" * 60)
print("DONE")
print("=" * 60)
