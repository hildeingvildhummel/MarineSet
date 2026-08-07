from HierarchicalKMeans import hierarchical_kmeans_with_resampling_streaming_version
import torch
import numpy as np
import os
import pickle
from support import sort_AISgekoppeld
import datetime
import faiss
from kmeans import OnlineKMeansFAISS

import argparse

def initiate_kmeans_dict(n_levels, pretrained=False):
    kmeans_dict = {}

    for t in range(n_levels):
        OnlineKmeans = OnlineKMeansFAISS(d=2048, k=n_clusters[t], use_gpu=True)

        if pretrained:
            with open("models/hkmeans/HKmeans_Update2_Backup21_restored/HKmeans_BigCluster_Update2_level{}_final.pkl".format(t),
                      "rb") as f:
                final_centroids = pickle.load(f)
            with open(
                    "models/hkmeans/HKmeans_Update2_Backup21_restored/HKmeans_BigCluster_Update2_level{}_initial.pkl".format(
                            t), "rb") as f:
                initial_centroids = pickle.load(f)
                n_cluster, dim = initial_centroids.shape

            final_kmeans = OnlineKMeansFAISS(d=dim, k=n_cluster, use_gpu=True)
            final_index = faiss.IndexFlatL2(dim)
            final_index.add(final_centroids.detach().numpy())

            # Assign to kmeans
            final_kmeans.index = final_index
            final_kmeans.centroids = final_centroids

            initial_kmeans = OnlineKMeansFAISS(d=dim, k=n_cluster, use_gpu=True)
            initial_index = faiss.IndexFlatL2(dim)
            initial_index.add(initial_centroids.detach().numpy())

            # Assign to kmeans
            initial_kmeans.index = initial_index
            initial_kmeans.centroids = initial_centroids

        else:
            initial_kmeans, initial_centroids = OnlineKmeans.initialize_kmeans()
            final_kmeans, final_centroids = OnlineKmeans.initialize_kmeans()

        sub_dict = {'initial': {'model': initial_kmeans, 'centroids': initial_centroids}, 'final': {'model': final_kmeans, 'centroids': final_centroids}}
        kmeans_dict[t] = sub_dict
    return kmeans_dict

def save_kmeans_models(kmeans_dict, save_base_name):
    if not os.path.exists('models/hkmeans/{}/'.format(save_base_name)):
        os.makedirs('models/hkmeans/{}/'.format(save_base_name))
    for level in kmeans_dict.keys():
        initial_kmeans = kmeans_dict[level]['initial']
        final_kmeans = kmeans_dict[level]['final']
        with open('models/hkmeans/{}/{}_level{}_initial.pkl'.format(save_base_name, save_base_name, str(level)), 'wb') as f:
            pickle.dump(initial_kmeans['centroids'], f)
        with open('models/hkmeans/{}/{}_level{}_final.pkl'.format(save_base_name, save_base_name, str(level)), 'wb') as f:
            pickle.dump(final_kmeans['centroids'], f)

def train_hkmeans_streamable(X, kmeans_dict, save_base_name):
    results = {}
    print('start training')
    res, kmeans_dict = hierarchical_kmeans_with_resampling_streaming_version(X, n_clusters, n_levels, sample_sizes, kmeans_dict)

    results[data_path] = res
    save_kmeans_models(kmeans_dict, save_base_name)
    return results, kmeans_dict

def load_saved_embeddings(data_path, start_index, max_size=5000):
    X = []
    time_stamps = []
    for index, file in enumerate(os.listdir(data_path)):
        try:
            start_time = datetime.datetime.strptime(('').join(file.split('.')[0].split('_')[-2:]), '%Y%m%d%H%M%S')
        except Exception as e:
            start_time = datetime.datetime.strptime(file.split('.')[0].split('_')[-1:][0], '%y%m%d%H%M%S')
        if index < start_index:
            continue
        embedding = np.load(os.path.join(data_path, file))
        total_sec = len(embedding) * 10
        end_time = start_time + datetime.timedelta(seconds=total_sec)
        current_time = start_time
        while current_time <= end_time:
            time_stamps.append(current_time)
            current_time += datetime.timedelta(seconds=10)
        X.extend(embedding)
        if np.array(X).shape[0]>= max_size:
            break
    return torch.from_numpy(np.array(X)), index, time_stamps


parser = argparse.ArgumentParser(description='Train hierarchical KMeans for automatic audio data curation')

parser.add_argument("--buck", help='The name of the bucket in Google Cloud Storage containing the raw audio.')
parser.add_argument('--dest', dest='dest', help='The local path to save the downloaded audio to.')
parser.add_argument('--save', dest='save', help='Base name to save the KMeans model to', required=True)


args = parser.parse_args()

if __name__ == "__main__":
    n_clusters = [6000, 400, 40, 10]
    n_levels = 4
    sample_sizes = [2200, 8, 5, 2]
    samplerate = 16000
    sample_size = 10
    bucket_name = "noaa-passive-bioacoustic"

    info = sort_AISgekoppeld('Administration/AISInformation.xlsx')
    destination_folder = args.dest

    kmeans_dict = initiate_kmeans_dict(n_levels, pretrained=True)

    keys = info.iloc[:, 0:3].agg("/".join, axis=1).to_list()

    for source_folder in keys:
        data_path = os.path.join(destination_folder, source_folder)
        if not os.path.isdir(data_path):
            continue
        total = len(os.listdir(data_path))
        start_index = 0
        while start_index < total:
            X, start_index, _ = load_saved_embeddings(data_path, start_index, max_size=n_clusters[0])
            results, kmeans_dict = train_hkmeans_streamable(X, kmeans_dict, args.save)
