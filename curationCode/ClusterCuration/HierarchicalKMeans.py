"""https://github.com/facebookresearch/ssl-data-curation/blob/main/src/hierarchical_kmeans_gpu.py"""
import sys
import logging
from tqdm import tqdm

import torch
import numpy as np

import kmeans as kmg

logger = logging.getLogger("hkmeans")
MEMORY_LIMIT = 1e8

def hierarchical_kmeans_with_resampling(
    data,
    n_clusters,
    n_levels,
    sample_sizes,
    n_resamples=10,
    init_method="kmeans++",
    num_init=1,
    sample_strategy="closest",
    verbose=True,
):
    """
    Run hierarchical k-means on data without resampling steps.

    Parameters:
        data: 2-D numpy array
            Data embeddings.
        n_clusters: List[int]
            Number of clusters for each level of hierarchical k-means
        n_levels: int
            Number of levels in hierarchical k-means.
        sample_size: List[int]
            Number of points to sample from each cluster in resampling steps.
        n_resamples: int
            Number of resampling steps in each level.
        init_method: str, default = "k-means++"
            Initialization method for k-means centroids.
            Options are "k-means" and "random".
        num_init: int, default=1
            Number of re-initialization for each k-means run.
        sampling_strategy: str, default = "closest"
            How to sample points from clusters in resampling steps.
            Options are "closest" and "random".

    Returns:
        List[dict], clustering results for each level of hierarchical k-means,
        including
            centroids: 2-D numpy array
                Centroids of clusters.
            assigment: 1-D numpy array
                Mapping from data points to cluster indices.
            clusters: array of array
            pot: float
                K-means potential.
    """
    assert len(n_clusters) == n_levels
    assert len(sample_sizes) == n_levels
    logger.info(f"{n_levels}-level hierarchical kmeans")
    res = []
    for kmid in range(n_levels):
        logger.info(f"Level {kmid+1}")
        logger.info("Initial kmeans")
        if kmid == 0:
            X = data
        else:
            X = res[kmid - 1]["centroids"]
        chunk_size = min(X.shape[0], int(MEMORY_LIMIT / n_clusters[kmid]))
        logger.info("Running the initial k-means")
        centroids, clusters, cluster_assignment, _ = kmg.kmeans(
            X,
            n_clusters=n_clusters[kmid],
            n_iters=50,
            chunk_size=chunk_size,
            num_init=num_init,
            init_method=init_method,
            dist="l2",
            high_precision=torch.float64,
            random_state=None,
            verbose=verbose,
        )
        logger.info("Resampling-kmeans")
        if sample_sizes[kmid] > 1:
            _sample_size = sample_sizes[kmid]
            for _ in tqdm(
                range(n_resamples),
                desc="Hierarchical k-means resampling steps",
                file=sys.stdout,
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                if sample_strategy == "closest":
                    sorted_clusters = [
                        _cluster[
                            torch.argsort(
                                torch.cdist(X[_cluster.astype(int)], centroids[i, None])
                                .flatten()
                            )
                            .cpu()
                            .numpy()
                        ]
                        for i, _cluster in enumerate(clusters)
                    ]
                    sampled_points = torch.concat(
                        [
                            X[_cluster.astype(int)[: _sample_size]]
                            for _cluster in sorted_clusters
                        ]
                    )
                elif sample_strategy == "random":
                    sampled_points = torch.concat(
                        [
                            X[
                                np.random.choice(
                                    _cluster,
                                    min(len(_cluster), _sample_size),
                                    replace=False
                                )
                            ]
                            for _cluster in clusters
                        ]
                    )
                else:
                    raise ValueError(
                        f"sample_strategy={sample_strategy} not supported!"
                    )
                chunk_size = min(
                    sampled_points.shape[0],
                    int(MEMORY_LIMIT / n_clusters[kmid])
                )
                centroids, _, _, _ = kmg.kmeans(
                    sampled_points,
                    n_clusters=n_clusters[kmid],
                    n_iters=50,
                    chunk_size=chunk_size,
                    num_init=num_init,
                    init_method=init_method,
                    dist="l2",
                    high_precision=torch.float64,
                    random_state=None,
                    verbose=False
                )
                cluster_assignment = kmg.assign_clusters(
                    centroids,
                    X,
                    "l2",
                    chunk_size=chunk_size,
                    verbose=False
                ).cpu().numpy()
                clusters = kmg.create_clusters_from_cluster_assignment(
                    cluster_assignment,
                    n_clusters[kmid]
                )
        res.append(
            {
                "centroids": centroids,
                "assignment": cluster_assignment,
                "clusters": clusters,
                "pot": -1,
            }
        )
    return res


def hierarchical_kmeans_with_resampling_streaming_version(
    data,
    n_clusters,
    n_levels,
    sample_sizes,
    kmeans_dict,
    n_resamples=10,
    init_method="kmeans++",
    num_init=1,
    sample_strategy="closest",
    verbose=True,
):
    """
    Run hierarchical k-means on data without resampling steps.

    Parameters:
        data: 2-D numpy array
            Data embeddings.
        n_clusters: List[int]
            Number of clusters for each level of hierarchical k-means
        n_levels: int
            Number of levels in hierarchical k-means.
        sample_size: List[int]
            Number of points to sample from each cluster in resampling steps.
        n_resamples: int
            Number of resampling steps in each level.
        init_method: str, default = "k-means++"
            Initialization method for k-means centroids.
            Options are "k-means" and "random".
        num_init: int, default=1
            Number of re-initialization for each k-means run.
        sampling_strategy: str, default = "closest"
            How to sample points from clusters in resampling steps.
            Options are "closest" and "random".

    Returns:
        List[dict], clustering results for each level of hierarchical k-means,
        including
            centroids: 2-D numpy array
                Centroids of clusters.
            assigment: 1-D numpy array
                Mapping from data points to cluster indices.
            clusters: array of array
            pot: float
                K-means potential.
    """
    assert len(n_clusters) == n_levels
    assert len(sample_sizes) == n_levels
    logger.info(f"{n_levels}-level hierarchical kmeans")
    res = []
    for kmid in range(n_levels):
        logger.info(f"Level {kmid+1}")
        logger.info("Initial kmeans")
        if kmid == 0:
            X = data
        else:
            X = res[kmid - 1]["centroids"]
        # chunk_size = min(X.shape[0], int(MEMORY_LIMIT / n_clusters[kmid]))
        logger.info("Running the initial k-means")
        initial_kmeans = kmeans_dict[kmid]['initial']
        # if kmid < 1:
        #     centroids, clusters, cluster_assignment, _, updated_initial_kmeans = kmg.streaming_kmeans(
        #         initial_kmeans, X, n_clusters[kmid], index_track
        #     )
        # else:
        print('initial kmeans', sample_strategy)
        centroids, clusters, cluster_assignment, _, updated_initial_kmeans = kmg.streaming_kmeans(
            initial_kmeans, X, n_clusters[kmid]
        )
        print('finished')
        kmeans_dict[kmid]['initial']['model'] = updated_initial_kmeans
        kmeans_dict[kmid]['initial']['centroids'] = centroids
        # centroids, clusters, cluster_assignment, _ = kmg.kmeans(
        #     X,
        #     n_clusters=n_clusters[kmid],
        #     n_iters=50,
        #     chunk_size=chunk_size,
        #     num_init=num_init,
        #     init_method=init_method,
        #     dist="l2",
        #     high_precision=torch.float64,
        #     random_state=None,
        #     verbose=verbose,
        # )
        logger.info("Resampling-kmeans")
        if sample_sizes[kmid] > 1:
            _sample_size = sample_sizes[kmid]
            for _ in tqdm(
                range(n_resamples),
                desc="Hierarchical k-means resampling steps",
                file=sys.stdout,
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                if sample_strategy == "closest":
                    sorted_clusters = [
                        _cluster[
                            torch.argsort(
                                torch.cdist(X[_cluster.astype(int)], centroids[i, None])
                                .flatten()
                            )
                            .cpu()
                            .numpy()
                        ]
                        for i, _cluster in enumerate(clusters)
                    ]
                    sampled_points = torch.concat([
                                             X[_cluster.astype(int)[: _sample_size]]
                                            for _cluster in sorted_clusters
                                            ])

                    # sampled_points_tuple = [(_cluster.astype(int)[: _sample_size],
                    #         X[_cluster.astype(int)[: _sample_size]])
                    #         for _cluster in sorted_clusters
                    #     ]
                    #
                    # sampled_points = torch.concat(
                    #     [x for _, x in sampled_points_tuple]
                    # )
                    # sampled_indices = [i for i, _ in sampled_points_tuple]
                    # sampled_index_track = index_track[np.concatenate(sampled_indices)]
                elif sample_strategy == "random":
                    sampled_points = torch.concat(
                        [
                            X[
                                np.random.choice(
                                    _cluster,
                                    min(len(_cluster), _sample_size),
                                    replace=False
                                )
                            ]
                            for _cluster in clusters
                        ]
                    )
                else:
                    raise ValueError(
                        f"sample_strategy={sample_strategy} not supported!"
                    )
                chunk_size = min(
                    sampled_points.shape[0],
                    int(MEMORY_LIMIT / n_clusters[kmid])
                )
                final_kmeans = kmeans_dict[kmid]['final']
                # centroids, _, _, _, updated_final_kmeans = kmg.streaming_kmeans(
                #     final_kmeans, sampled_points, sampled_index_track, n_clusters[kmid]
                # )
                # if kmid < 1:
                #     centroids, _, _, _, updated_final_kmeans = kmg.streaming_kmeans(
                #         final_kmeans, sampled_points, n_clusters[kmid], sampled_index_track
                #     )
                # else:
                centroids, _, _, _, updated_final_kmeans = kmg.streaming_kmeans(
                    final_kmeans, sampled_points, n_clusters[kmid]
                )

                cluster_assignment = kmg.assign_clusters(
                    centroids,
                    X,
                    "l2",
                    chunk_size=chunk_size,
                    verbose=False
                ).cpu().numpy()
                # clusters = kmg.create_clusters_from_cluster_assignment_streaming_version(
                #     cluster_assignment,
                #     sampled_index_track
                # )
                # if kmid < 1:
                #     clusters = kmg.create_clusters_from_cluster_assignment_streaming_version(
                #         cluster_assignment,
                #         index_track, n_clusters[kmid]
                #     )
                # else:
                clusters = kmg.create_clusters_from_cluster_assignment(cluster_assignment, n_clusters[kmid])
            kmeans_dict[kmid]['final']['model'] = updated_final_kmeans
            kmeans_dict[kmid]['final']['centroids'] = centroids
            # index_track = sampled_index_track
        res.append(
            {
                "centroids": centroids,
                "assignment": cluster_assignment,
                "clusters": clusters,
                "pot": -1,
            }
        )
    return res, kmeans_dict


def hierarchical_kmeans_application(
    data,
    n_clusters,
    n_levels,
    sample_sizes,
    kmeans_dict,
):
    """
    Run hierarchical k-means on data without resampling steps.

    Parameters:
        data: 2-D numpy array
            Data embeddings.
        n_clusters: List[int]
            Number of clusters for each level of hierarchical k-means
        n_levels: int
            Number of levels in hierarchical k-means.
        sample_size: List[int]
            Number of points to sample from each cluster in resampling steps.
        n_resamples: int
            Number of resampling steps in each level.
        init_method: str, default = "k-means++"
            Initialization method for k-means centroids.
            Options are "k-means" and "random".
        num_init: int, default=1
            Number of re-initialization for each k-means run.
        sampling_strategy: str, default = "closest"
            How to sample points from clusters in resampling steps.
            Options are "closest" and "random".

    Returns:
        List[dict], clustering results for each level of hierarchical k-means,
        including
            centroids: 2-D numpy array
                Centroids of clusters.
            assigment: 1-D numpy array
                Mapping from data points to cluster indices.
            clusters: array of array
            pot: float
                K-means potential.
    """
    assert len(n_clusters) == n_levels
    assert len(sample_sizes) == n_levels
    logger.info(f"{n_levels}-level hierarchical kmeans")
    res = []
    for kmid in range(n_levels):
        if kmid == 0:
            X = data
        else:
            X = res[kmid - 1]["centroids"]
        # print('Selected data: ', kmid, X)
        chunk_size = min(
            X.shape[0],
            int(MEMORY_LIMIT / n_clusters[kmid])
        )
        final_kmeans = kmeans_dict[kmid]['final']
        centroids, clusters, cluster_assignment, score = kmg.streaming_kmeans_application(
            final_kmeans, X, n_clusters[kmid]
        )
        cluster_assignment = kmg.assign_clusters(
            centroids,
            X,
            "l2",
            chunk_size=chunk_size,
            verbose=False
        ).cpu().numpy()
        clusters = kmg.create_clusters_from_cluster_assignment(
            cluster_assignment,
            n_clusters[kmid]
        )
        res.append(
            {
                "centroids": centroids,
                "assignment": cluster_assignment,
                "clusters": clusters,
                "pot": score,
            }
        )
    return res