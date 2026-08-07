import os
import gc
import yaml
import pickle
import argparse
import numpy as np
import torch

from Download_Selection import sampled_curation
from support import sort_AISgekoppeld, clean_folder
from Conformer_Embeddings import load_conformer_model, Conformer
from GoogleCloudConnection import download_folder
from AudioCuration_TrainHKmeans import initiate_kmeans_dict, load_saved_embeddings, train_hkmeans_streamable
from AudioCuration_Sampling import resample_from_trained_hkmeans


# --------------------------------------------------
# ARGPARSE
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical KMeans audio curation pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--dest", required=True, help="Local download directory.")
    parser.add_argument("--save", required=True, help="Model save name.")
    return parser.parse_args()


# --------------------------------------------------
# CONFIG LOADER
# --------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# SAFE DIRECTORY CREATION
# --------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    # ---------------- CONFIG ----------------
    bucket_name = config["bucket_name"]
    samplerate = config["audio"]["samplerate"]
    sample_size = config["audio"]["sample_size"]

    n_clusters = config["hkmeans"]["n_clusters"]
    n_levels = config["hkmeans"]["n_levels"]
    sample_sizes = config["hkmeans"]["sample_sizes"]
    N = config["hkmeans"]["N_total"]

    ais_path = config["paths"]["ais_info"]
    embedding_root = config["paths"]["embedding_save_root"]
    hkmeans_root = config["paths"]["hkmeans_model_root"]
    curated_root = config["paths"]["curated_root"]
    final_output = config["paths"]["final_curation_output"]
    temp_output = config["paths"]["temp_output"]

    destination_folder = args.dest

    # Ensure required directories exist
    ensure_dir(destination_folder)
    ensure_dir(embedding_root)
    ensure_dir(hkmeans_root)
    ensure_dir(curated_root)
    ensure_dir(final_output)
    ensure_dir(temp_output)

    # ==================================================
    # 1️⃣ DOWNLOAD + EMBEDDING EXTRACTION
    # ==================================================
    print("Starting embedding extraction...")

    info = sort_AISgekoppeld(ais_path)
    keys = info.iloc[:, 0:3].agg("/".join, axis=1).to_list()
    audio_keys = [key + "/audio" for key in keys]

    model = load_conformer_model()

    for source_folder in audio_keys:
        print("Processing:", source_folder)

        total = float("inf")
        start_index = 0

        while start_index < total:
            start_index, total = download_folder(
                bucket_name,
                source_folder,
                destination_folder,
                start_index=start_index,
            )

            data_path = os.path.join(destination_folder, source_folder)
            ensure_dir(data_path)

            embedding_model = Conformer(
                model, data_path, samplerate, sample_size
            )

            try:
                X = embedding_model()
            except Exception as e:
                print("Embedding failed:", str(e))
                clean_folder(destination_folder)
                continue

            save_dir = os.path.join(embedding_root, source_folder)
            ensure_dir(os.path.dirname(save_dir))

            files = sorted(os.listdir(data_path))
            if not files:
                continue

            start_name = files[0].split(".")[0]
            end_name = files[-1].split(".")[0]

            npy_path = f"{save_dir}_{start_name}_{end_name}.npy"

            with open(npy_path, "wb") as f:
                np.save(f, X.detach().cpu().numpy())

            clean_folder(destination_folder)

            del X
            del embedding_model
            gc.collect()
            torch.cuda.empty_cache()

    # ==================================================
    # 2️⃣ TRAIN HIERARCHICAL KMEANS
    # ==================================================
    print("Starting hierarchical KMeans training...")

    kmeans_dict = initiate_kmeans_dict(n_levels, pretrained=True)

    for source_folder in keys:
        data_path = os.path.join(destination_folder, source_folder)

        if not os.path.isdir(data_path):
            continue

        total = len(os.listdir(data_path))
        start_index = 0

        while start_index < total:
            X, start_index, _ = load_saved_embeddings(
                data_path,
                start_index,
                max_size=n_clusters[0],
            )

            _, kmeans_dict = train_hkmeans_streamable(
                X,
                kmeans_dict,
                args.save,
            )

    # ==================================================
    # 3️⃣ RESAMPLING FROM TRAINED HKMEANS
    # ==================================================
    print("Starting resampling...")

    curated_path = os.path.join(
        curated_root, f"Curated_{args.save}_MarineSet.pkl"
    )

    ensure_dir(curated_root)

    with open(curated_path, "rb") as f:
        resampled = pickle.load(f)

    for source_folder in keys:
        data_path = os.path.join(destination_folder, source_folder)

        if not os.path.isdir(data_path):
            continue

        total = len(os.listdir(data_path))
        start_index = 0

        while start_index < total - 1:
            resampled, start_index = resample_from_trained_hkmeans(
                os.path.join(hkmeans_root, args.save),
                data_path,
                n_clusters,
                sample_sizes,
                N,
                resampled,
                start_index,
            )

            with open(curated_path, "wb") as f:
                pickle.dump(resampled, f)

    # ==================================================
    # 4️⃣ FINAL CURATION
    # ==================================================
    print("Running final sampled curation...")

    sampled_curation(
        curated_path,
        curation_save_path=final_output,
        temp_save_path=temp_output,
    )


if __name__ == "__main__":
    main()
