"""https://github.com/facebookresearch/ssl-data-curation/blob/main/src/kmeans_gpu.py"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import torch
from tqdm import tqdm

from sklearn.utils import check_random_state
import torch


# def create_clusters_from_cluster_assignment(
#     cluster_assignment: np.array,
#     num_clusters: int,
#     return_object_array: bool = True,
# ):
#     """
#     Build clusters from cluster assignment.
#     """
#     ID = np.argsort(cluster_assignment)
#     print('ID: ', ID)
#     sorted_cluster_assigment = cluster_assignment[ID]
#     print('sorted: ', sorted_cluster_assigment)
#     index_split = np.searchsorted(sorted_cluster_assigment, list(range(num_clusters)))
#     print('index split: ', index_split)
#     clusters = np.split(ID, index_split[1:])
#     print('clusters: ', clusters)
#     if return_object_array:
#         return np.array(clusters, dtype=object)
#     else:
#         return clusters

def create_clusters_from_cluster_assignment(assignments, n_clusters):
    return [np.where(assignments == i)[0] for i in range(n_clusters)]

def create_clusters_from_cluster_assignment_streaming_version(
    cluster_assignment: np.array,
    indices_track: np.array,
    num_clusters: int,
    return_object_array: bool = True,
):
    """
    Build clusters from cluster assignment.
    """
    ID = np.argsort(cluster_assignment)
    sorted_cluster_assigment = cluster_assignment[ID]
    sorted_index_track = indices_track[ID]
    index_split = np.searchsorted(sorted_cluster_assigment, list(range(num_clusters)))
    clusters = np.split(sorted_index_track, index_split[1:])
    if return_object_array:
        return np.array(clusters, dtype=object)
    else:
        return clusters

def matmul_transpose(X, Y):
    """
    Compute X . Y.T
    """
    return torch.matmul(X, Y.T)


def compute_distance(
    X, Y, Y_squared_norms, dist="l2", X_squared_norm=None, matmul_fn=matmul_transpose
):
    """
    Compute pairwise distance between rows of X and Y.

    Parameters:
        X: torch.tensor of shape (n_samples_x, n_features)
        Y: torch.tensor of shape (n_samples_y, n_features)
            Y is supposed to be larger than X.
        Y_squared_norms: torch.tensor of shape (n_samples_y, )
            Squared L2 norm of rows of Y.
            It can be  provided to avoid re-computation.
        dist: 'cos' or 'l2'
            If 'cos', assuming that rows of X are normalized
            to have L2 norm equal to 1.
        X_squared_norm: torch.tensor of shape (n_samples_x, )
            Squared L2 norm of rows of X.
        matmul_fn: matmul function.

    Returns:

        Pairwise distance between rows of X and Y.

    """

    if dist == "cos":
        return 2 - 2 * matmul_fn(X, Y)
    elif dist == "l2":
        if X_squared_norm is None:
            X_squared_norm = torch.linalg.vector_norm(X, dim=1) ** 2
        return X_squared_norm[:, None] - 2 * matmul_fn(X, Y) + Y_squared_norms[None, :]
    else:
        raise ValueError(f'dist = "{dist}" not supported!')


# A modified version of _kmeans_plusplus
# from https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/cluster/_kmeans.py#L63
def kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    dist,
    random_state=None,
    n_local_trials=None,
    high_precision=torch.float64,
    verbose=False,
):
    """
    Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
        X : torch.tensor of shape (n_samples, n_features)
            The data to pick seeds for.
        n_clusters : int
            The number of seeds to choose.
        x_squared_norms : torch.tensor (n_samples,)
            Squared Euclidean norm of each data point.
        random_state : RandomState instance
            The generator used to initialize the centers.
        n_local_trials : int, default=None
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.
        high_precision: torch.float32 or torch.float64, to save GPU memory, one
            can use float32 or float16 for data 'X', 'high_precision' will be
            use in aggregation operation to avoid overflow.

    Returns
        centers : torch.tensor of shape (n_clusters, n_features)
            The initial centers for k-means.
        indices : ndarray of shape (n_clusters,)
            The index location of the chosen centers in the data array X. For a
            given index and center, X[index] = center.

    """
    if random_state is None:
        random_state = check_random_state(random_state)

    n_samples, n_features = X.shape

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype).to(X.device)
    pots = torch.empty((n_clusters,), device=X.device, dtype=high_precision)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = compute_distance(X[center_id, None], X, x_squared_norms, dist)[
        0
    ].type(high_precision)
    current_pot = closest_dist_sq.sum()
    pots[0] = current_pot

    # Pick the remaining n_clusters-1 points
    if verbose:
        iterates = tqdm(
            range(1, n_clusters),
            desc="Kmeans++ initialization",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(1, n_clusters)
    for c in iterates:
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = (
            torch.tensor(random_state.uniform(size=n_local_trials)).to(
                current_pot.device
            )
            * current_pot
        )
        candidate_ids = torch.searchsorted(
            torch.cumsum(closest_dist_sq, dim=0), rand_vals
        )
        # numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, None, closest_dist_sq.shape[0] - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = compute_distance(
            X[candidate_ids], X, x_squared_norms, dist
        ).type(high_precision)

        # update closest distances squared and potential for each candidate
        torch.minimum(
            closest_dist_sq, distance_to_candidates, out=distance_to_candidates
        )
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
        pots[c] = current_pot

    return centers, indices


def assign_clusters(centroids, X, dist, chunk_size=-1, verbose=False):
    """
    Assign data points to their closest clusters.

    Parameters:

        centroids: torch.tensor of shape (n_clusters, n_features)
            Centroids of the clusters.
        X: torch.tensor of shape (n_samples, n_features)
            Data.
        dist: 'cos' or 'l2'
            If 'cos', assuming that rows of X are normalized
            to have L2 norm equal to 1.
        chunk_size: int
            Number of data points that are assigned at once.
            Use a small chunk_size if n_clusters is large to avoid
            out-of-memory error, e.g. chunk_size <= 1e9/n_clusters.
            Default is -1, meaning all data points are assigned at once.
        verbose: bool
            Whether to print progress bar.

    Returns:

        torch.tensor of shape (n_samples, ) containing the cluster id of
        each data point.

    """

    cluster_ids = []
    n_samples, _ = X.shape
    x_squared_norms = torch.linalg.vector_norm(X, dim=1) ** 2
    centroid_squared_norm = torch.linalg.vector_norm(centroids, dim=1) ** 2
    if chunk_size < 0:
        try:
            distance_from_centroids = compute_distance(
                centroids, X, x_squared_norms, dist, centroid_squared_norm
            )
        except Exception as e:
            raise MemoryError(
                f"matrices are too large, consider setting chunk_size ({chunk_size}) to a smaller number"
            ) from e
        cluster_ids = torch.argmin(distance_from_centroids, dim=0)
    else:
        n_iters = (n_samples + chunk_size - 1) // chunk_size
        if verbose:
            iterates = tqdm(
                range(n_iters),
                desc="Assigning data points to centroids",
                file=sys.stdout,
                bar_format="{l_bar}{bar}{r_bar}",
            )
        else:
            iterates = range(n_iters)

        for chunk_idx in iterates:
            begin_idx = chunk_idx * chunk_size
            end_idx = min(n_samples, (chunk_idx + 1) * chunk_size)
            distance_from_centroids = compute_distance(
                centroids,
                X[begin_idx:end_idx],
                x_squared_norms[begin_idx:end_idx],
                dist,
                centroid_squared_norm,
            )
            cluster_ids.append(torch.argmin(distance_from_centroids, dim=0))
            del distance_from_centroids
        cluster_ids = torch.cat(cluster_ids)
    return cluster_ids


def compute_centroids(
    centroids, cluster_assignment, n_clusters, X, high_precision=torch.float32
):
    """
    Compute centroids of each cluster given its data points.

    Parameters:

        centroids: torch.tensor of shape (n_clusters, n_features)
            Previous centroids of the clusters.
        cluster_assignment: torch.tensor of shape (n_samples, )
            Cluster id of data points.
        n_clusters: int
            Number of clusters.
        X: torch.tensor of shape (n_samples, n_features)
            Data.
        high_precision: torch.float32 or torch.float64, to save GPU memory, one
            can use float32 or float16 for data 'X', 'high_precision' will be
            use in aggregation operation to avoid overflow.

    Returns:

        torch.tensor of shape (n_clusters, n_features), new centroids
    """
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    new_centroids = torch.zeros_like(centroids)
    for i in range(n_clusters):
        if len(clusters[i]) > 0:
            new_centroids[i] = torch.mean(
                X[clusters[i].astype(int)].type(high_precision), dim=0
            )
        else:
            new_centroids[i] = centroids[i]
    return new_centroids


def _kmeans(
    X,
    n_clusters,
    n_iters,
    chunk_size=-1,
    init_method="kmeans++",
    dist="l2",
    high_precision=torch.float32,
    random_state=None,
    verbose=False,
):
    """
    Run kmeans once.

    Parameters: See above.

    Returns:

        centroids:
        clusters: np.array of np.array
            Indices of points in each cluster. A subarray corresponds to a cluster.
        cluster_assignment:
        pot: float, kmeans objective

    """
    if random_state is None:
        random_state = check_random_state(random_state)

    x_squared_norms = torch.linalg.vector_norm(X, dim=1) ** 2
    if init_method == "kmeans++":
        centroids, _ = kmeans_plusplus(
            X,
            n_clusters,
            x_squared_norms,
            dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        centroids = torch.tensor(
            X[np.sort(random_state.choice(range(len(X)), n_clusters, replace=False))],
            device=X.device,
            dtype=X.dtype,
        )

    cluster_assignment = assign_clusters(centroids, X, dist, chunk_size).cpu().numpy()
    for _iter in range(n_iters):
        centroids = compute_centroids(
            centroids, cluster_assignment, n_clusters, X, high_precision
        )
        cluster_assignment = (
            assign_clusters(centroids, X, dist, chunk_size).cpu().numpy()
        )
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    pot = np.sum(
        [
            torch.sum(
                torch.cdist(
                    X[el.astype(int)], X[el.astype(int)].mean(dim=0, keepdim=True)
                )
                ** 2
            ).item()
            for el in clusters
        ]
    )
    return centroids, clusters, cluster_assignment, pot


def kmeans(
    X,
    n_clusters,
    n_iters,
    chunk_size=-1,
    num_init=10,
    init_method="kmeans++",
    dist="l2",
    high_precision=torch.float32,
    random_state=None,
    verbose=False,
):
    """
    Run kmeans multiple times and return the clustering with the best objective.

    Parameters: See above and

        num_init: int
            Number of kmeans runs.

    Returns:

        Same as _kmeans

    """

    n_clusters = min(X.shape[0], n_clusters)
    best_centroids, best_clusters, best_cluster_assignment, best_pot = (
        None,
        None,
        None,
        np.Inf,
    )
    for _ in range(num_init):
        centroids, clusters, cluster_assignment, pot = _kmeans(
            X,
            n_clusters,
            n_iters,
            chunk_size=chunk_size,
            init_method=init_method,
            dist=dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
        if pot < best_pot:
            best_centroids, best_clusters, best_cluster_assignment, best_pot = (
                centroids,
                clusters,
                cluster_assignment,
                pot,
            )
    return best_centroids, best_clusters, best_cluster_assignment, best_pot

def streaming_clusters(cluster_assignment):
    cluster_assignment = cluster_assignment.detach().numpy()
    clusters = []
    for i in range(np.min(cluster_assignment), np.max(cluster_assignment)+1):
        indices = np.where(cluster_assignment == i)
        clusters.append(indices[0].tolist())
    return np.array(clusters)

def streaming_kmeans(kmeans_model, X, num_clusters, indices_track = None, num_init=10):
    # n_clusters = min(X.shape[0], n_clusters)
    # OnlineKMeans = Faiss_Dask(k=num_clusters, d=2048)
    OnlineKMeans = OnlineKMeansFAISS(k=num_clusters, d=2048)
    X = X.detach().numpy()
    # best_centroids, best_clusters, best_cluster_assignment, best_pot = (
    #     None,
    #     None,
    #     None,
    #     np.inf,
    # )
    # kmeans_model, centroids = OnlineKMeans.load(kmeans_model)
    # kmeans, centroids = OnlineKMeans.update_centroids(X, centroids)
    kmeans_model, centroids = OnlineKMeans.load_kmeans(kmeans_model['model'])
    kmeans_model.update(X.astype(np.float32))
    centroids = kmeans_model._get_centroids(kmeans_model)
    kmeans = kmeans_model
    # kmeans, centroids = OnlineKMeans.update_centroids(X, kmeans_model['centroids'])
    # kmeans, centroids = OnlineKMeans.train(X, kmeans_model)
    # pot = kmeans.score(X)
    # centroids = kmeans.centroids
    # pot = np.mean(kmeans.obj)
    # print(np.mean(pot))

    # centroids = torch.from_numpy(kmeans.cluster_centers_)
    # centroids = torch.from_numpy(centroids)
    # cluster_assignment = kmeans.predict(X)
    D, cluster_assignment = kmeans.index.search(X, 1)
    # D, cluster_assignment = OnlineKMeans.cluster_assignment(X, kmeans)
    pot = np.mean(D)
    if indices_track is None:
        clusters = create_clusters_from_cluster_assignment(cluster_assignment.flatten(), num_clusters)
    else:
        clusters = create_clusters_from_cluster_assignment_streaming_version(cluster_assignment.flatten(),
                                                                             indices_track, num_clusters)

    # centroids, clusters, cluster_assignment, pot = _kmeans(
    #     X,
    #     n_clusters,
    #     n_iters,
    #     chunk_size=chunk_size,
    #     init_method=init_method,
    #     dist=dist,
    #     high_precision=high_precision,
    #     random_state=random_state,
    #     verbose=verbose,
    # )
    # if pot < best_pot:
    #     best_centroids, best_clusters, best_cluster_assignment, best_pot, best_kmeans = (
    #         centroids,
    #         clusters,
    #         cluster_assignment,
    #         pot,
    #         kmeans
    #     )
    best_centroids, best_clusters, best_cluster_assignment, best_pot, best_kmeans = (
        centroids,
        clusters,
        cluster_assignment,
        pot,
        kmeans
    )
    # # n_clusters = min(X.shape[0], n_clusters)
    # OnlineKMeans = OnlineKMeansFAISS(k=num_clusters, d= 2048)
    # X = X.detach().numpy()
    # # best_centroids, best_clusters, best_cluster_assignment, best_pot = (
    # #     None,
    # #     None,
    # #     None,
    # #     np.inf,
    # # )
    # kmeans_model = OnlineKMeans.load_kmeans(kmeans_model)
    # # print('N iterations: ', num_init)
    # # for _ in range(num_init):
    #     # kmeans = kmeans_model.partial_fit(X)
    #     # try:
    #     #     kmeans_model = OnlineKMeans.load_kmeans(kmeans_model)
    #     #     kmeans_model.update(X)
    #     #     centroids = kmeans_model.centroids
    #     # except:
    #     #     kmeans, centroids = OnlineKMeans.fit(X, kmeans_model)
    # kmeans = OnlineKMeans.fit(X, kmeans_model)
    # print('fitted')
    # # pot = kmeans.score(X)
    # centroids = kmeans.centroids
    # pot = np.mean(kmeans.obj)
    # print(np.mean(pot))
    #
    # # centroids = torch.from_numpy(kmeans.cluster_centers_)
    # centroids = torch.from_numpy(centroids)
    # # cluster_assignment = kmeans.predict(X)
    # D, cluster_assignment = kmeans.index.search(X, 1)
    # print(cluster_assignment.flatten())
    # if indices_track is None:
    #     clusters = create_clusters_from_cluster_assignment(cluster_assignment.flatten(), num_clusters)
    # else:
    #     clusters = create_clusters_from_cluster_assignment_streaming_version(cluster_assignment.flatten(), indices_track, num_clusters)
    #
    # # centroids, clusters, cluster_assignment, pot = _kmeans(
    # #     X,
    # #     n_clusters,
    # #     n_iters,
    # #     chunk_size=chunk_size,
    # #     init_method=init_method,
    # #     dist=dist,
    # #     high_precision=high_precision,
    # #     random_state=random_state,
    # #     verbose=verbose,
    # # )
    # # if pot < best_pot:
    # #     best_centroids, best_clusters, best_cluster_assignment, best_pot, best_kmeans = (
    # #         centroids,
    # #         clusters,
    # #         cluster_assignment,
    # #         pot,
    # #         kmeans
    # #     )
    # best_centroids, best_clusters, best_cluster_assignment, best_pot, best_kmeans = (
    #     centroids,
    #     clusters,
    #     cluster_assignment,
    #     pot,
    #     kmeans
    # )
    return best_centroids, best_clusters, best_cluster_assignment, best_pot, best_kmeans


def streaming_kmeans_application(kmeans_model, X, n_clusters, use_gpu=True):
    centroids = kmeans_model
    # Move to GPU if available and requested
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    X, centroids = X.to(device), centroids.to(device)

    # Compute squared L2 distance using efficient broadcasting
    # dist(X, C) = ||X - C||^2 = ||X||^2 + ||C||^2 - 2 * X @ C.T
    X_norm = (X ** 2).sum(dim=1, keepdim=True)  # Shape (N, 1)
    C_norm = (centroids ** 2).sum(dim=1, keepdim=True).T  # Shape (1, K)
    distances = X_norm + C_norm - 2 * (X @ centroids.T)  # Shape (N, K)

    # Assign each point to the nearest centroid
    cluster_assignment = torch.argmin(distances, dim=1)  # Shape (N,)
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)

    # Compute the mean distance to the assigned centroids
    min_distances = torch.min(distances, dim=1)[0]
    mean_distance = min_distances.mean().item()

    return centroids, clusters, cluster_assignment, mean_distance
    # # Ensure X is on GPU
    # if X.is_cuda:
    #     X = X.cpu()  # FAISS requires NumPy
    # X = X.detach().numpy().astype('float32')  # Convert to NumPy for FAISS
    #
    # centroids = kmeans_model
    #
    # dimension = centroids.shape[1]
    #
    # if use_gpu:
    #     # Initialize FAISS GPU resources
    #     res = faiss.StandardGpuResources()
    #
    #     # Create a GPU index for L2 search
    #     index = faiss.IndexFlatL2(dimension)  # Create CPU index
    #     index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
    # else:
    #     index = faiss.IndexFlatL2(dimension)  # CPU version
    #
    # # Step 2: Add centroids to the index
    # index.add(centroids)  # The index now contains the centroids
    #
    # # Step 3: Search for the nearest centroid for each data point
    # distances, cluster_assignment = index.search(X, 1)  # FAISS runs on GPU now
    #
    # # Compute the potential (mean distance)
    # pot = np.mean(distances)
    #
    # # Convert outputs back to PyTorch tensors on GPU if needed
    # cluster_assignment = torch.tensor(cluster_assignment, dtype=torch.long, device='cuda' if use_gpu else 'cpu')
    #
    # clusters = create_clusters_from_cluster_assignment(cluster_assignment.flatten(), n_clusters)
    #
    # # print(centroids.shape, clusters.shape, cluster_assignment.flatten().shape, pot)
    #
    # return centroids, clusters, cluster_assignment, pot


# def streaming_kmeans_application(kmeans_model, X, n_clusters):
#     X = X.detach().numpy()
#
#     centroids = kmeans_model
#
#     # kmeans = kmeans_model
#     # pot = kmeans.score(X)
#
#     dimension = centroids.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # Create L2 (Euclidean) index
#
#     # Step 2: Add centroids to the index
#     index.add(centroids)  # The index now contains the centroids
#
#     # Step 3: Search for the nearest centroid for each data point
#     distances, cluster_assignment = index.search(X, 1)
#     pot = np.mean(distances)
#
#     # centroids = torch.from_numpy(kmeans.cluster_centers_)
#     # cluster_assignment = kmeans.predict(X)
#     clusters = create_clusters_from_cluster_assignment(cluster_assignment.flatten(), n_clusters)
#     print(centroids.shape, clusters.shape, cluster_assignment.flatten().shape, pot)
#     return centroids, clusters, cluster_assignment, pot


def sort_cluster_by_distance(
    X, centroids, clusters, device="cuda", dtype=torch.float32, verbose=False,
):
    """
    Sort data points in each cluster in increasing order of distance to the centroid.

    Parameters:

        X: data
        centroids:
        clusters:

    Returns:

        sorted_clusters: np.array of np.array
            Indices of points in each cluster. A subarray corresponds to a cluster.

    """

    n_clusters, n_dim = centroids.shape[0], centroids.shape[1]

    sorted_clusters = []
    if verbose:
        iterates = tqdm(
            range(n_clusters),
            desc="Sorting clusters by distance",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(n_clusters)
    for cluster_idx in iterates:
        if len(clusters[cluster_idx]) > 0:
            point_indices = np.sort(clusters[cluster_idx]).astype(int)
            point_feats = torch.tensor(X[point_indices], device=device, dtype=dtype)
            _centroid = centroids[cluster_idx].reshape(1, n_dim).type(dtype)

            dist_to_centroid = torch.cdist(point_feats, _centroid).flatten()
            sorted_clusters.append(
                point_indices[torch.argsort(dist_to_centroid).cpu().numpy()]
            )
            del point_feats
        else:
            sorted_clusters.append(np.array([]).astype(int))
    return np.array(sorted_clusters, dtype=object)

import numpy as np
import faiss


class OnlineKMeansFAISS:
    def __init__(self, k=500000, alpha=0.1, d=128, use_gpu=False):
        """
        k: Number of clusters
        alpha: Learning rate (controls how much new data influences the centroid update)
        d: Dimension of data points
        use_gpu: Whether to use GPU for acceleration
        nlist: Number of cluster groups for FAISS IVF indexing
        """
        self.k = k
        self.alpha = alpha
        self.d = d
        self.use_gpu = use_gpu
        # self.nlist = nlist  # Number of cluster partitions (for IVF index)

        self.centroids = None
        self.counts = None  # Track number of points per cluster

        # # # Create an IVF index for fast searching
        # quantizer = faiss.IndexFlatL2(d)  # Coarse quantizer
        # self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # if use_gpu:
        #     self.res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)  # Move to GPU

    def initialize_kmeans(self):
        kmeans = faiss.Kmeans(d=self.d, k=self.k, niter=10, verbose=True, gpu=self.use_gpu)
        centroids = kmeans.centroids
        return kmeans, centroids

    def load_kmeans(self, kmeans):
        return kmeans, kmeans.centroids
    def fit(self, X, kmeans):
        """Initial training using FAISS KMeans"""

        kmeans.train(X.astype(np.float32))  # FAISS expects float32

        # # self.centroids = kmeans.centroids
        # self.counts = np.ones(self.k)  # Start counts to avoid division by zero
        #
        # self.index.train(self.centroids.astype(np.float32))  # Train FAISS IVF index
        # self.index.add(self.centroids.astype(np.float32))  # Add initial centroids
        # return kmeans, self.centroids
        return kmeans, kmeans.centroids

    def _get_centroids(self, kmeans):
        self.centroids = kmeans.centroids
        return self.centroids

    def _assign_clusters(self, X):
        """Finds the nearest centroid for each data point using FAISS"""
        _, labels = self.index.search(X.astype(np.float32), 1)  # Fast nearest neighbor search
        return labels.flatten()

    def update(self, new_data):
        """Online incremental update using multiple data streams"""
        labels = self._assign_clusters(new_data)

        if self.counts is None:
            self.counts = np.ones(self.k)

        for i, point in zip(labels, new_data):
            self.counts[i] += 1
            eta = 1 / self.counts[i]  # Adaptive learning rate (less impact from later data)
            self.centroids[i] = (1 - eta) * self.centroids[i] + eta * point

        # Update FAISS index dynamically (without resetting)
        self.index.reset()
        self.index.add(self.centroids.cpu().numpy().astype(np.float32))

import numpy as np
import faiss
# from dask.distributed import Client, LocalCluster
# import dask.array as da
import os

class Faiss_Dask():
    def __init__(self, d=2048, k=500000, batch_size=100000, num_batches=100, lr=0.1):

        # ---- PARAMETERS ----
        self.d = d  # Feature dimension
        self.k = k  # Number of clusters
        self.buffer_size = k  # Reservoir sampling buffer size
        self.batch_size = batch_size  # Streaming batch size per worker
        self.num_batches = num_batches  # Total streaming updates
        self.num_gpus = faiss.get_num_gpus()  # Number of GPUs
        self.lr = lr  # Learning rate for updating centroids

        # ---- 1. SETUP DASK FOR MULTI-GPU PROCESSING ----
        cluster = LocalCluster(n_workers=self.num_gpus, threads_per_worker=1, address="localhost:40000")  # One worker per GPU
        self.client = Client(cluster)
        print("✅ Dask Cluster Started:", self.client)

    def initialize(self):
        # ---- 2. INITIALIZE FAISS INDEX FOR KMEANS ----
        res = [faiss.StandardGpuResources() for _ in range(self.num_gpus)]  # GPU resources
        self.gpu_indices = [
            faiss.index_cpu_to_gpu(res[i], i, faiss.IndexFlatL2(self.d))  # GPU-based L2 search
            for i in range(self.num_gpus)
        ]

        # ---- 3. INITIALIZE RESERVOIR SAMPLING BUFFER ----
        self.reservoir = np.random.rand(self.buffer_size, self.d).astype('float32')  # Initial dataset
        self.centroids = self.reservoir[np.random.choice(self.buffer_size, self.k, replace=False)]  # Random init

        print(f"✅` Initialized centroids with {self.buffer_size} samples.")
        return self.gpu_indices, self.centroids

    def load(self, dictionary):
        indices = dictionary['model']
        centroids = dictionary['centroids']
        return indices, centroids

    # ---- 4. FUNCTION TO UPDATE CENTROIDS (DISTRIBUTED) ----
    def update_centroids(self, batch, centroids):
        """
        Update cluster centroids using Stochastic Gradient Descent (SGD) on GPUs.
        """
        batch_size, d = batch.shape
        batch_gpu = da.from_array(batch, chunks=(batch_size // self.num_gpus, d))  # Dask parallel batch
        results = []

        def process_batch(batch_chunk, gpu_idx):
            distances, cluster_ids = self.gpu_indices[gpu_idx].search(batch_chunk, 1)  # GPU-based search
            for i in range(len(batch_chunk)):
                cluster_id = cluster_ids[i][0]
                centroids[cluster_id] = (1 - self.lr) * centroids[cluster_id] + self.lr * batch_chunk[i]
            return centroids

        # Parallel execution across GPUs
        for i in range(self.num_gpus):
            results.append(self.client.submit(process_batch, batch_gpu[i].compute(), i))

        # Aggregate results
        self.centroids = self.client.gather(results)[0]
        self.save_model('models/hkmeans/', self.centroids, self.gpu_indices[0])
        return self.gpu_indices, self.centroids

    def stream_data(self):
        # ---- 5. STREAM DATA AND UPDATE CENTROIDS ----
        for step in range(self.num_batches):
            new_batch = np.random.rand(self.batch_size, self.d).astype('float32')  # Simulated streaming data

            # Reservoir Sampling: Replace old points with new data
            replace_idx = np.random.choice(self.buffer_size, self.batch_size, replace=False)
            self.reservoir[replace_idx] = new_batch

            # Parallel centroid update
            self.centroids = self.update_centroids(centroids, new_batch, self.lr)

            # Adaptive learning rate decay
            self.lr *= 0.99

            print(f"🔄 Step {step + 1}: Updated centroids with {self.batch_size} new samples (lr={lr:.5f})")

    def cluster_assignment(self, query_data, index):
        # ---- 6. QUERY CLUSTER ASSIGNMENTS ----
        #     query_data = np.random.rand(5, self.d).astype('float32')  # Sample queries
        D, I = index.search(query_data, 1)  # Query using GPU-accelerated FAISS

        print("🔍 Query Results (Nearest Clusters):", I)
        return D, I

    def shutdown_dask(self):
        # ---- 7. SHUTDOWN DASK ----
        self.client.close()
        print("✅ Dask Cluster Shut Down.")

    def save_model(self, model_dir, centroids, index, filename1="centroids.npy", filename2="faiss_index.index"):
        np.save(os.path.join(model_dir, filename1), centroids)  # Save centroids to .npy file
        print(f"✅ Saved centroids to {filename1}")
        faiss.write_index(index, os.path.join(model_dir, filename2))  # Save FAISS index
        print(f"✅ Saved FAISS index to {filename2}")

    def load_saved_model(self, model_dir, filename1="centroids.npy", filename2="faiss_index.index"):
        centroids = np.load(os.path.join(model_dir, filename1))
        index = faiss.read_index(os.path.join(model_dir, filename2))
        return centroids, index

# class OnlineKMeansFAISS:
#     def __init__(self, k=3, alpha=0.1, d=2):
#         """
#         k: Number of clusters
#         alpha: Learning rate (controls how much new data influences the centroid update)
#         d: Dimension of data points
#         """
#         self.k = k
#         self.alpha = alpha
#         self.d = d
#         self.centroids = None
#         self.counts = None  # Number of points assigned to each cluster
#         self.index = faiss.IndexFlatL2(d)  # Fast L2 distance search index
#
#     def initialize_kmeans(self):
#         kmeans = faiss.Kmeans(d=self.d, k=self.k, niter=10, verbose=False, gpu=True)
#         return kmeans
#
#     def load_kmeans(self, kmeans):
#         return kmeans
#
#     def fit(self, X, kmeans):
#         """Initialize centroids using FAISS KMeans"""
#         kmeans.train(X.astype(np.float32))  # FAISS expects float32
#         self.centroids = kmeans.centroids  # Get centroids from FAISS
#         self.counts = np.zeros(self.k)  # Initialize cluster counts
#         self.index.add(self.centroids)  # Add centroids to FAISS index for fast search
#         return kmeans, self.centroids
#
#     def _assign_clusters(self, X):
#         """Finds the nearest centroid for each data point using FAISS"""
#         _, labels = self.index.search(X.astype(np.float32), 1)  # Search nearest centroid
#         return labels.flatten()
#
#     def update(self, new_data):
#         """Online update of centroids using new data"""
#         labels = self._assign_clusters(new_data)
#
#         for i, point in zip(labels, new_data):
#             self.counts[i] += 1
#             # Update centroid using a weighted moving average
#             self.centroids[i] = (1 - self.alpha) * self.centroids[i] + self.alpha * point
#
#         # Update FAISS index
#         self.index = faiss.IndexFlatL2(self.d)  # Reset FAISS index
#         self.index.add(self.centroids.astype(np.float32))  # Add updated centroids