"""Before running this script for the first time locally, first apply the steps provided in
https://googleapis.dev/python/google-api-core/latest/auth.html"""

import os
import pickle
import argparse
import yaml

from AISdata import AISDataCuration
from AISSelection import AISAudioExtractor


# ==============================================================
# Helpers
# ==============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================
# Args
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AIS → Audio extraction pipeline"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )

    # Optional overrides
    parser.add_argument("--t", type=int, help="Override curation t")
    parser.add_argument("--run-curation", action="store_true")

    return parser.parse_args()


# ==============================================================
# Main
# ==============================================================

def main():

    args = parse_args()
    cfg = load_config(args.config)

    # ----------------------------------------------------------
    # Read config
    # ----------------------------------------------------------

    t = args.t if args.t else cfg["curation"]["t"]
    run_curation = args.run_curation or cfg["curation"]["run"]

    ais_folder = cfg["curation"]["ais_folder"]

    info_excel = cfg["paths"]["info_excel"]
    selection_dir = cfg["paths"]["selection_dir"]
    temp_dir = cfg["paths"]["temp_dir"]
    output_dir = cfg["paths"]["output_dir"]

    bucket = cfg["cloud"]["bucket_name"]

    curated_pickle = os.path.join(selection_dir, f"curatedAIS_{t}t.pkl")

    # ----------------------------------------------------------
    # Ensure folders exist
    # ----------------------------------------------------------

    for folder in [selection_dir, temp_dir, output_dir]:
        ensure_dir(folder)

    # ----------------------------------------------------------
    # Step 1 — AIS curation
    # ----------------------------------------------------------

    if run_curation or not os.path.exists(curated_pickle):
        print(f"Running AIS curation (t={t})")

        curator = AISDataCuration(AIS_folder=ais_folder, t=t)
        D_star = curator()

        with open(curated_pickle, "wb") as f:
            pickle.dump(D_star, f)

        print("Saved →", curated_pickle)
    else:
        print("Using existing pickle →", curated_pickle)

    # ----------------------------------------------------------
    # Step 2 — extraction
    # ----------------------------------------------------------

    extractor = AISAudioExtractor(
        info_excel=info_excel,
        curated_pickle=curated_pickle,
        temp_dir=temp_dir,
        output_dir=output_dir,
        bucket_name=bucket,
    )

    extractor.process()

    print("Done ✓")
    os.remove(temp_dir)


# ==============================================================
# Entry
# ==============================================================

if __name__ == "__main__":
    main()
