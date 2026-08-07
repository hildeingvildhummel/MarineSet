from HierarchicalKMeans import hierarchical_kmeans_application
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
import pickle
from clusters import HierarchicalCluster
from hierarchical_sampling import hierarchical_sampling
import pandas as pd
from support import sort_AISgekoppeld
import datetime
import argparse

def initiate_kmeans_dict(n_levels):
    MEMORY_LIMIT = 1e8
    kmeans_dict = {}
    for t in range(n_levels):
        chunk_size = int(MEMORY_LIMIT / n_clusters[t])
        initial_kmeans = MiniBatchKMeans(n_clusters=n_clusters[t],
                                         random_state=0,
                                         batch_size=chunk_size,
                                         max_iter=50,
                                         init='k-means++')
        final_kmeans = MiniBatchKMeans(n_clusters=n_clusters[t],
                                         random_state=0,
                                         batch_size=chunk_size,
                                         max_iter=50,
                                         init='k-means++')
        sub_dict = {'initial': initial_kmeans, 'final': final_kmeans}
        kmeans_dict[t] = sub_dict
    return kmeans_dict
def merge_two_dictionaries(d1, d2):
    for index, (dict1, dict2) in enumerate(zip(d1, d2)):
        assigned_clusters1 = dict1['clusters']
        assigned_clusters2 = dict2['clusters']
        counter_index = 0
        for cluster1, cluster2 in zip(assigned_clusters1, assigned_clusters2):
            cluster1 = np.concatenate((cluster1, cluster2), axis=0)
            dict1['clusters'][counter_index] = cluster1
            counter_index += 1
        dict1['assignment'] = np.concatenate((dict1['assignment'], dict2['assignment']), axis=0)
        dict1['pot'] += dict2['pot']
        d1[index] = dict1
    return d1

def get_kmeans_clusterAssignment(kmeans_file_base_folder):
    kmeans_dict = {}
    for file in os.listdir(kmeans_file_base_folder):
        kmeans_level = int(file.split('_')[-2][-1])
        type_kmeans = file.split('_')[-1].split('.')[0]
        basename = args.save
        kmeans_filename = basename+'_level'+str(kmeans_level)+'_'+type_kmeans+'.pkl'
        kmeans_file = os.path.join(kmeans_file_base_folder, kmeans_filename)
        with open(kmeans_file, 'rb') as f:
            kmeans = pickle.load(f)
        if kmeans_level in kmeans_dict.keys():
            kmeans_dict[kmeans_level][type_kmeans] = kmeans
        else:
            kmeans_dict[kmeans_level] = {}
            kmeans_dict[kmeans_level][type_kmeans] = kmeans
    return kmeans_dict

def update_sample_dict(sample_dict, new_distances, timestamps, data_path, N):
    sample_dict['distance'] = torch.cat((sample_dict['distance'], new_distances), 0)
    sample_dict['timestamps'] = np.concatenate((sample_dict['timestamps'], timestamps), 0)
    sample_dict['path'] = np.concatenate((sample_dict['path'], np.repeat(data_path, len(new_distances))), 0)
    if len(sample_dict['timestamps']) > N:
        sorted_indices = torch.argsort(sample_dict["distance"])
        # Sort all dictionary entries using the same indices
        sorted_data = {
            key: (value[sorted_indices] if isinstance(value, torch.Tensor) else value[sorted_indices.numpy()])
            for key, value in sample_dict.items()
        }
        sample_dict = {
            key: value[:N] for key, value in sorted_data.items()
        }
        print([[key, value.shape] for key, value in sample_dict.items()])
    return sample_dict


def map_distance_sampled_datapoints(X, sampled_indices, centroids, assignment):
    from torch import nn
    pdist = nn.PairwiseDistance(p=2)
    datapoints = X[sampled_indices]
    counter = 0
    for level, cluster_id in zip(centroids, assignment):
        sub_clusters = cluster_id[sampled_indices]
        sub_centroid = level[sub_clusters]
        if counter == 0:
            distance = pdist(datapoints, sub_centroid)
            counter += 1
        else:
            added_distance = pdist(datapoints, sub_centroid)
            distance = torch.add(distance, added_distance)
            counter += 1
        datapoints = sub_centroid
        sampled_indices = sub_clusters
    return distance

def get_AIS_start_and_end_times(df):
    """Optimized function to extract start and end times of AIS occurrences."""

    # Convert to datetime (faster if already in correct format)
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')

    # Ensure data is sorted
    df.sort_values(by=["MMSI", "BaseDateTime"], inplace=True)

    # Compute time differences using NumPy for speed
    df['Time_Diff'] = np.concatenate([[np.nan], np.diff(df['BaseDateTime'].astype('int64')) / 1e9])

    # Identify new occurrences based on threshold
    threshold = 600  # 10 minutes
    df['Group_ID'] = (df['Time_Diff'] > threshold).cumsum()

    # Aggregate start and end times for each MMSI occurrence
    result = df.groupby(["MMSI", "Group_ID"], as_index=False).agg(
        Start_Time=("BaseDateTime", "min"),
        End_Time=("BaseDateTime", "max")
    ).drop(columns=["Group_ID"])
    return result

def remove_AIS_sample(X, data_path, timestamps):
    """Optimized function to remove AIS samples based on timestamps."""

    # Load AIS metadata
    info = sort_AISgekoppeld('Administratie/AISgekoppeld_2023.xlsx')

    base_path = os.path.basename(os.path.dirname(data_path))
    df_new = info[info['Unnamed: 2'] == base_path]

    # Extract AIS file names and time ranges
    AIS_info = df_new['AIS data file'].tolist()

    # Preload all AIS data once (instead of reading inside the loop)
    AIS_data = {}
    for AIS_file in AIS_info:
        # AIS_path = os.path.join('/mnt/d/CWI/Data/AIS2023/', AIS_file)
        AIS_path = os.path.join('/projects/0/vusr0637/AIS2023/', AIS_file)
        AIS_data[AIS_file] = get_AIS_start_and_end_times(pd.read_csv(AIS_path))

    # Find indices to remove
    to_remove = []
    timestamps = pd.to_datetime(timestamps)  # Ensure timestamps are in datetime format
    for idx, selected_time in enumerate(timestamps):
        for AIS_file, times in AIS_data.items():
            mask = (selected_time >= times['Start_Time']) & (selected_time <= times['End_Time'])
            if mask.any():
                to_remove.append(idx)
                break  # No need to check further if a match is found
    # Create and apply mask
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    mask[to_remove] = False
    return X[mask]

def resample_from_trained_hkmeans(kmeans_file_base_folder, data_path, n_clusters, sample_sizes, N, resampled,
                                  start_index,sub_target=10):
    # resampled = None
    kmeans_dict = get_kmeans_clusterAssignment(kmeans_file_base_folder)
    prev_cluster = None

    X , index, time_stamps = load_saved_embeddings(data_path=data_path, start_index=start_index, max_size=50000)

    """Remove AIS samples here"""
    X = remove_AIS_sample(X, data_path, time_stamps)
    clusters = hierarchical_kmeans_application(X, n_clusters, len(kmeans_dict.keys()), sample_sizes, kmeans_dict)
    if prev_cluster is not None:
        prev_cluster = merge_two_dictionaries(clusters, prev_cluster)
    else:
        prev_cluster = clusters
    cl = HierarchicalCluster.from_dict(prev_cluster)
    sampled_indices = hierarchical_sampling(cl, target_size=N, sampling_strategy='c')
    distance_values = map_distance_sampled_datapoints(X, sampled_indices, [x['centroids'] for x in prev_cluster], [x['assignment'] for x in prev_cluster])
    if resampled is None:
        resampled = {'distance': distance_values,
                     'timestamps': np.array(time_stamps)[sampled_indices],
                     'path': np.repeat(data_path, len(sampled_indices))}
    else:
        resampled = update_sample_dict(resampled, distance_values, time_stamps, data_path, N)
    return resampled, index
def load_saved_embeddings(data_path, start_index, max_size=5000):
    X = []
    time_stamps = []
    for index, file in enumerate(os.listdir(data_path)):
        try:
            start_time = datetime.datetime.strptime(('').join(file.split('.')[0].split('_')[-2:]), '%Y%m%d%H%M%S')
        except:
            try:
                start_time = datetime.datetime.strptime(('').join(file.split('.')[0].split('_')[-2:]), '%y%m%d%H%M%S')
            except:
                start_time = datetime.datetime.strptime(('').join(file.split('.')[0].split('_')[-1:]), '%y%m%d%H%M%S')
        if index < start_index:
            continue
        try:
            embedding = np.load(os.path.join(data_path, file))
        except:
            continue
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
    N = 1080000 - 348553

    info = sort_AISgekoppeld('Administration/AISInformation.xlsx')
    destination_folder = args.dest

    kmeans_dict = initiate_kmeans_dict(n_levels)

    keys = info.iloc[:, 0:3].agg("/".join, axis=1).to_list()

    with open('Curated/Curated_{}_MarineSet.pkl'.format(args.save), 'rb') as f:
        resampled = pickle.load(f)

    for source_folder in keys:
        print(source_folder)
        data_path = os.path.join(destination_folder, source_folder)
        if not os.path.isdir(data_path):
            print('continue')
            continue

        total = len(os.listdir(data_path))
        print('Total: ', total)
        start_index = 0
        while start_index < total-1:
            resampled, start_index = resample_from_trained_hkmeans('models/hkmeans/{}/'.format(args.save), data_path,
                                                                   n_clusters, sample_sizes,
                                                      N, resampled, start_index)
            print(start_index)
            with open('Curated/Curated_{}_MarineSet.pkl'.format(args.save), 'wb') as f:
                pickle.dump(resampled, f)
