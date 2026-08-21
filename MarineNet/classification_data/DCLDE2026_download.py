import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from google.cloud import storage


# ============================================================
# Google Cloud configuration
# ============================================================

os.environ["GCLOUD_PROJECT"] = "noaa-passive-bioacoustic"

BUCKET_NAME = "noaa-passive-bioacoustic"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


# ============================================================
# Configuration
# ============================================================

ANNOTATION_FILE = "Annotations.csv"

BUCKET_PREFIX = (
    "dclde/2027/"
    "dclde_2027_killer_whales"
)

# Use a NEW directory so old 10-second windows are not reused
OUTPUT_DIR = Path(
    "/scratch-shared/hhummel/DCLDE2027_windows_5s"
)

TEMP_DIR = Path(
    "/scratch-shared/hhummel/DCLDE2027_temp"
)

WINDOW_LENGTH = 5.0
TARGET_SR = 16000

NUM_WORKERS = 1

MULTILABEL_THRESHOLD = 0.3

RANDOM_SEED = 42


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LOG_FILE = OUTPUT_DIR / "processing.log"

METADATA_FILE = OUTPUT_DIR / "windows.csv"


# ============================================================
# Logging
# ============================================================

def log(message):

    with open(
        LOG_FILE,
        "a",
        buffering=1,
    ) as f:

        f.write(
            message + "\n"
        )

        f.flush()

    print(
        message,
        flush=True,
    )


# ============================================================
# Load annotations
# ============================================================

df = pd.read_csv(
    ANNOTATION_FILE
)

df = df[
    (df["FileOk"] == True)
    &
    df["ClassSpecies"].notna()
].copy()

df["ClassSpecies"] = (
    df["ClassSpecies"]
    .astype(str)
)

df = df[
    df["ClassSpecies"]
    .str
    .strip()
    != ""
].copy()


print("=" * 60)
print("DCLDE 2027")
print("=" * 60)

print(
    f"Annotations:       "
    f"{len(df):,}"
)

print(
    f"Recordings:        "
    f"{df['FilePath'].nunique():,}"
)

print(
    f"Classes:           "
    f"{df['ClassSpecies'].nunique():,}"
)

print("\nClass distribution:")

print(
    df["ClassSpecies"]
    .value_counts()
)

print("=" * 60)


# ============================================================
# Google Cloud Storage path
# ============================================================

def get_remote_path(row):

    file_path = (
        str(row["FilePath"])
        .replace("\\", "/")
    )

    relative_path = file_path.split(
        "DCLDE/",
        1
    )[1]

    # Lowercase directories only
    directory = relative_path.rsplit(
        "/",
        1
    )[0].lower()

    # Preserve original filename
    filename = str(
        row["Soundfile"]
    )

    return (
        f"{BUCKET_PREFIX}/"
        f"{directory}/"
        f"{filename}"
    )


# ============================================================
# Label cleaning
# ============================================================

def clean_label(label):

    return (
        str(label)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ============================================================
# Deterministic random generator
# ============================================================

def get_annotation_rng(annotation_index):
    """
    Create a deterministic random generator for each annotation.

    This ensures that the same annotation always receives the
    same random window, even after restarting the script.
    """

    return np.random.default_rng(
        RANDOM_SEED + int(annotation_index)
    )


# ============================================================
# Random window generation
# ============================================================

def get_random_window(
    event_start,
    event_end,
    recording_duration,
    rng,
):
    """
    Create a WINDOW_LENGTH-second window containing the target
    annotation.

    For annotations shorter than the window, the annotation is
    randomly positioned between:

        - starting at the beginning of the window
        - ending at the end of the window

    For annotations longer than the window, a random
    WINDOW_LENGTH-second segment inside the annotation is used.

    The window is shifted if necessary to remain inside the
    recording.
    """

    event_length = (
        event_end - event_start
    )

    if event_length <= 0:

        raise ValueError(
            f"Invalid annotation: "
            f"{event_start} - {event_end}"
        )


    # --------------------------------------------------------
    # Short annotation
    # --------------------------------------------------------

    if event_length <= WINDOW_LENGTH:

        max_offset = (
            WINDOW_LENGTH
            - event_length
        )

        offset = rng.uniform(
            0,
            max_offset,
        )

        window_start = (
            event_start
            - offset
        )


    # --------------------------------------------------------
    # Long annotation
    # --------------------------------------------------------

    else:

        window_start = rng.uniform(
            event_start,
            event_end
            - WINDOW_LENGTH,
        )


    window_end = (
        window_start
        + WINDOW_LENGTH
    )


    # --------------------------------------------------------
    # Shift to recording boundaries
    # --------------------------------------------------------

    if recording_duration >= WINDOW_LENGTH:

        if window_start < 0:

            window_start = 0.0

            window_end = (
                WINDOW_LENGTH
            )

        elif window_end > recording_duration:

            window_end = (
                recording_duration
            )

            window_start = (
                recording_duration
                - WINDOW_LENGTH
            )


    # --------------------------------------------------------
    # Extremely short recordings
    # --------------------------------------------------------

    else:

        window_start = 0.0

        window_end = (
            recording_duration
        )


    return (
        window_start,
        window_end,
    )


# ============================================================
# Find relevant annotations
# ============================================================

def get_relevant_annotations(
    annotations,
    window_start,
    window_end,
):
    """
    Return annotations satisfying:

        intersection
        ----------------------------- > MULTILABEL_THRESHOLD
        min(window_length, annotation_length)

    The target annotation will later always be included,
    regardless of numerical edge cases.
    """

    relevant_indices = []


    for idx, annotation in annotations.iterrows():

        annotation_start = float(
            annotation[
                "FileBeginSec"
            ]
        )

        annotation_end = float(
            annotation[
                "FileEndSec"
            ]
        )

        annotation_length = (
            annotation_end
            - annotation_start
        )


        if annotation_length <= 0:
            continue


        # Intersection duration
        intersection = max(
            0.0,

            min(
                window_end,
                annotation_end,
            )

            -

            max(
                window_start,
                annotation_start,
            ),
        )


        denominator = min(
            WINDOW_LENGTH,
            annotation_length,
        )


        overlap_ratio = (
            intersection
            / denominator
        )


        if (
            overlap_ratio
            > MULTILABEL_THRESHOLD
        ):

            relevant_indices.append(
                idx
            )


    return annotations.loc[
        relevant_indices
    ]


# ============================================================
# Determine annotation output
# ============================================================

def get_annotation_info(
    annotation_index,
    row,
    annotations,
    recording_duration,
):
    """
    Determine the deterministic random window, relevant labels,
    multilabel status, and output path for one annotation.
    """

    event_start = float(
        row["FileBeginSec"]
    )

    event_end = float(
        row["FileEndSec"]
    )


    # --------------------------------------------------------
    # Deterministic RNG
    # --------------------------------------------------------

    rng = get_annotation_rng(
        annotation_index
    )


    # --------------------------------------------------------
    # Random window
    # --------------------------------------------------------

    window_start, window_end = (
        get_random_window(
            event_start=event_start,
            event_end=event_end,
            recording_duration=recording_duration,
            rng=rng,
        )
    )


    # --------------------------------------------------------
    # Relevant annotations
    # --------------------------------------------------------

    relevant_annotations = (
        get_relevant_annotations(
            annotations=annotations,
            window_start=window_start,
            window_end=window_end,
        )
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = sorted(
        relevant_annotations[
            "ClassSpecies"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    # Always include target annotation
    target_label = str(
        row["ClassSpecies"]
    )


    if target_label not in labels:

        labels.append(
            target_label
        )


    labels = sorted(
        labels
    )


    is_multilabel = (
        len(labels) > 1
    )


    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    output_filename = (
        f"{Path(row['Soundfile']).stem}"
        f"_ann_{annotation_index}.wav"
    )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if is_multilabel:

        output_dir = (
            OUTPUT_DIR
            / "multilabel"
        )

    else:

        output_dir = (
            OUTPUT_DIR
            / clean_label(
                target_label
            )
        )


    output_path = (
        output_dir
        / output_filename
    )


    return {

        "output_path": output_path,

        "event_start": event_start,

        "event_end": event_end,

        "window_start": window_start,

        "window_end": window_end,

        "labels": labels,

        "is_multilabel": is_multilabel,

        "relevant_annotations":
            relevant_annotations,

    }


# ============================================================
# Check whether recording is already processed
# ============================================================

def recording_already_processed(
    annotations,
    recording_duration,
):
    """
    Return True only if all expected windows already exist.
    """

    for annotation_index, row in annotations.iterrows():

        info = get_annotation_info(
            annotation_index=annotation_index,
            row=row,
            annotations=annotations,
            recording_duration=recording_duration,
        )


        if not info[
            "output_path"
        ].exists():

            return False


    return True


# ============================================================
# Generate metadata for existing windows
# ============================================================

def generate_metadata_for_recording(
    annotations,
    recording_duration,
    source_sr=None,
):
    """
    Generate metadata for windows that already exist.
    """

    metadata = []


    for annotation_index, row in annotations.iterrows():

        info = get_annotation_info(
            annotation_index=annotation_index,
            row=row,
            annotations=annotations,
            recording_duration=recording_duration,
        )


        output_path = info[
            "output_path"
        ]


        if not output_path.exists():
            continue


        metadata.append({

            "file": str(
                output_path.relative_to(
                    OUTPUT_DIR
                )
            ),

            "primary_label":
                clean_label(
                    row["ClassSpecies"]
                ),

            "labels":
                "|".join(
                    info["labels"]
                ),

            "multilabel":
                info["is_multilabel"],

            "recording":
                row["Soundfile"],

            "dataset":
                row["Dataset"],

            "original_filepath":
                row["FilePath"],

            "window_start":
                info["window_start"],

            "window_end":
                info["window_end"],

            "window_length":
                (
                        info["window_end"]
                        - info["window_start"]
                ),

            "annotation_start":
                info["event_start"],

            "annotation_end":
                info["event_end"],

            "num_annotations":
                len(
                    info[
                        "relevant_annotations"
                    ]
                ),

            "num_labels":
                len(
                    info["labels"]
                ),

            "original_sample_rate":
                source_sr,

            "sample_rate":
                TARGET_SR,

        })


    return metadata


# ============================================================
# Process one source recording
# ============================================================

def process_recording(group):

    file_path, annotations = group

    first_row = annotations.iloc[0]

    filename = Path(
        first_row["Soundfile"]
    ).name

    local_audio = (
        TEMP_DIR
        / filename
    )


    try:

        # ====================================================
        # Download source recording
        # ====================================================

        blob_name = get_remote_path(
            first_row
        )

        blob = bucket.blob(
            blob_name
        )

        blob.download_to_filename(
            str(local_audio),
            timeout=300,
        )


        # ====================================================
        # Read recording information
        # ====================================================

        audio_info = sf.info(
            local_audio
        )

        source_sr = (
            audio_info.samplerate
        )

        recording_duration = (
            audio_info.frames
            / source_sr
        )


        # ====================================================
        # Check whether all windows already exist
        # ====================================================

        if recording_already_processed(
            annotations=annotations,
            recording_duration=recording_duration,
        ):

            metadata = (
                generate_metadata_for_recording(
                    annotations=annotations,
                    recording_duration=
                        recording_duration,
                    source_sr=source_sr,
                )
            )


            local_audio.unlink(
                missing_ok=True
            )


            return {

                "success": True,

                "skipped": True,

                "file": filename,

                "error": None,

                "metadata": metadata,

            }


        # ====================================================
        # Process annotations
        # ====================================================

        metadata = []


        for annotation_index, row in annotations.iterrows():

            info = get_annotation_info(
                annotation_index=
                    annotation_index,

                row=row,

                annotations=
                    annotations,

                recording_duration=
                    recording_duration,
            )


            output_path = (
                info["output_path"]
            )


            # ------------------------------------------------
            # Skip existing window
            # ------------------------------------------------

            if output_path.exists():

                metadata.extend(

                    generate_metadata_for_recording(
                        annotations=
                            annotations.loc[
                                [annotation_index]
                            ],

                        recording_duration=
                            recording_duration,

                        source_sr=
                            source_sr,
                    )

                )

                continue

            # ------------------------------------------------
            # Read source segment
            # ------------------------------------------------

            read_start = max(
                0.0,
                info["window_start"],
            )

            read_end = min(
                recording_duration,
                info["window_end"],
            )

            start_frame = int(
                read_start * source_sr
            )

            num_frames = int(
                (read_end - read_start)
                * source_sr
            )

            audio, _ = sf.read(
                local_audio,
                start=start_frame,
                frames=num_frames,
                always_2d=False,
            )

            # ------------------------------------------------
            # Convert stereo -> mono
            # ------------------------------------------------

            if audio.ndim > 1:
                audio = np.mean(
                    audio,
                    axis=1,
                )

            # ------------------------------------------------
            # Resample to target sample rate
            # ------------------------------------------------

            if source_sr != TARGET_SR:
                audio = resample_poly(
                    audio,
                    TARGET_SR,
                    source_sr,
                )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            sf.write(
                output_path,
                audio,
                TARGET_SR,
            )


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata.append({

                "file": str(
                    output_path.relative_to(
                        OUTPUT_DIR
                    )
                ),

                "primary_label":
                    clean_label(
                        row["ClassSpecies"]
                    ),

                "labels":
                    "|".join(
                        info["labels"]
                    ),

                "multilabel":
                    info["is_multilabel"],

                "recording":
                    row["Soundfile"],

                "dataset":
                    row["Dataset"],

                "original_filepath":
                    row["FilePath"],

                "window_start":
                    info["window_start"],

                "window_end":
                    info["window_end"],

                "window_length":
                    (
                            info["window_end"]
                            - info["window_start"]
                    ),

                "annotation_start":
                    info["event_start"],

                "annotation_end":
                    info["event_end"],

                "num_annotations":
                    len(
                        info[
                            "relevant_annotations"
                        ]
                    ),

                "num_labels":
                    len(
                        info["labels"]
                    ),

                "original_sample_rate":
                    source_sr,

                "sample_rate":
                    TARGET_SR,

            })


        # ====================================================
        # Remove temporary recording
        # ====================================================

        local_audio.unlink(
            missing_ok=True
        )


        return {

            "success": True,

            "skipped": False,

            "file": filename,

            "error": None,

            "metadata": metadata,

        }


    except Exception as e:

        local_audio.unlink(
            missing_ok=True
        )


        error = (
            f"{type(e).__name__}: {e}"
        )


        log(
            f"FAILED | "
            f"{filename} | "
            f"{error}"
        )


        return {

            "success": False,

            "skipped": False,

            "file": filename,

            "error": error,

            "metadata": [],

        }


# ============================================================
# Create recording groups
# ============================================================

groups = list(
    df.groupby(
        "FilePath"
    )
)

# ------------------------------------------------------------
# TEST MODE
# Only process the first N unique recordings
# ------------------------------------------------------------

TEST_MODE = False
N_TEST_RECORDINGS = 10

if TEST_MODE:
    groups = groups[:N_TEST_RECORDINGS]

    print(
        f"\nTEST MODE enabled"
    )

    print(
        f"Processing first "
        f"{len(groups)} "
        f"unique recordings"
    )

else:

    print(
        f"\nUnique recordings to process: "
        f"{len(groups):,}"
    )

print(
    f"Using {NUM_WORKERS} "
    f"parallel workers\n"
)


# ============================================================
# Load existing metadata
# ============================================================

if METADATA_FILE.exists():

    existing_metadata = pd.read_csv(
        METADATA_FILE
    )

    existing_files = set(
        existing_metadata["file"]
        .astype(str)
    )

    print(
        f"Existing metadata rows: "
        f"{len(existing_metadata):,}"
    )

else:

    existing_files = set()


# ============================================================
# Process recordings
# ============================================================

failed = []


with ThreadPoolExecutor(
    max_workers=NUM_WORKERS
) as executor:

    futures = [

        executor.submit(
            process_recording,
            group,
        )

        for group in groups

    ]


    for future in tqdm(

        as_completed(futures),

        total=len(futures),

        desc="Downloading/extracting",

    ):

        result = future.result()


        if result["success"]:

            if result["metadata"]:

                # Avoid duplicate metadata
                new_metadata = [

                    row

                    for row in result["metadata"]

                    if row["file"]
                    not in existing_files

                ]


                if new_metadata:

                    result_df = pd.DataFrame(
                        new_metadata
                    )


                    write_header = (
                        not METADATA_FILE.exists()
                    )


                    result_df.to_csv(

                        METADATA_FILE,

                        mode="a",

                        header=write_header,

                        index=False,

                    )


                    existing_files.update(

                        row["file"]

                        for row
                        in new_metadata

                    )


        else:

            failed.append(
                result
            )


# ============================================================
# Save failed recordings
# ============================================================

if failed:

    failed_df = pd.DataFrame(
        failed
    )

    failed_df.to_csv(
        OUTPUT_DIR
        / "failed_recordings.csv",
        index=False,
    )


# ============================================================
# Reload metadata for summary
# ============================================================

if METADATA_FILE.exists():

    metadata_df = pd.read_csv(
        METADATA_FILE
    )

else:

    metadata_df = pd.DataFrame()


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 60)
print("FINISHED")
print("=" * 60)

print(
    f"Windows extracted: "
    f"{len(metadata_df):,}"
)


if len(metadata_df):

    n_multi = int(
        metadata_df[
            "multilabel"
        ].sum()
    )


    print(
        f"Single-label windows: "
        f"{len(metadata_df) - n_multi:,}"
    )

    print(
        f"Multi-label windows: "
        f"{n_multi:,}"
    )


    print(
        "\nSingle-label distribution:"
    )


    print(

        metadata_df[
            ~metadata_df[
                "multilabel"
            ]
        ][
            "primary_label"
        ].value_counts()

    )


print(
    f"\nMetadata:\n"
    f"{METADATA_FILE}"
)


if failed:

    print(
        f"\nFailed recordings: "
        f"{len(failed):,}"
    )

    print(
        f"See:\n"
        f"{OUTPUT_DIR / 'failed_recordings.csv'}"
    )