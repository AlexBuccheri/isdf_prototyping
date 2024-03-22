""" Module for functions that find interpolation grid points:

* QR decomposition with column pivoting
* Weighted k-means clustering
"""
from typing import List

import numpy as np
from scipy.spatial.distance import cdist


cluster_type = List[List[int]]

def assign_points_to_centroids(grid_points: np.ndarray, centroids: np.ndarray) -> cluster_type:
    """

    :return: clusters: List of length N interpolation points. Each element corresponds to a cluster,
    which indexes all grid points associated with it
    """
    N_r, dim = grid_points.shape
    N_interp = centroids.shape[0]
    clusters: cluster_type = [[]] * N_interp

    # Computing distance matrix per grid point because I want to distribute this loop
    for ir in range(0, N_r):
        distance_matrix = cdist(grid_points[ir, :], centroids)
        # min(|r_{ir} - centroids|)
        # TODO, this is probs a 2D index - will want the column
        icen = np.argmin(distance_matrix)
        clusters[icen].append(ir)

    return clusters


def update_centroids(grid_points, f_weight, clusters: cluster_type) -> np.ndarray:
    """ Compute a new set of centroids

    Have as many clusters as we do centroids

    :param grid_points:
    :param f_weight:
    :param indices:
    :return:
    """
    N_interp = len(clusters)
    dim = grid_points.shape[1]
    updated_centroids = np.empty(shape=(N_interp, dim))

    for icen in range(0, N_interp):
        # Indices of grid points associated with this cluster
        grid_indices = np.asarray(clusters[icen])
        # Part of the weight function associated with this cluster
        weights = f_weight[grid_indices]
        weighted_pos = np.sum(grid_points[grid_indices] * weights)
        updated_centroids[icen, :] = weighted_pos / np.sum(weights)
    return updated_centroids



def points_are_converged(updated_points, points, tol, verbose=False) -> bool:
    """ Given the difference in two sets of points, determine whether the updated
    points are sufficiently close to the prior points.

    :param updated_points:
    :param points:
    :return:
    """
    vector_diffs = updated_points - points
    norm = np.linalg.norm(vector_diffs)
    indices = np.where(norm > tol)[0]
    converged = len(indices) == 0

    if verbose:
        N = updated_points.shape[0]
        if converged:
            print(f'Convergence: All points converged')
        else:
            print(f'Convergence: {len(indices)} points out of {N} are not converged:')
            print('# Current Point    Prior Point    |ri - r_{i-1}|   tol')
            for i in indices:
                print(updated_points[i, :], points[i, :], norm[i], tol

    return converged


# See below for refactor
#     # This should be refactored with linear algebra
#     for i, centroid in enumerate(centroids):
#         centroid_in_grid = False
#         for grid_point in grid_points:
#             diff = centroid - grid_point
#             if np.linalg.norm(diff) <= tol:
#                 centroid_in_grid = True
#                 break
#         if not centroid_in_grid:
#             raise ValueError(f'Centroid point {centroid} with index {i} was not found in real-space grid')


def is_subgrid_on_grid(subgrid, grid, tol) -> np.ndarray:
    """ Return indices of subgrid points not present in grid.

    :param subgrid:
    :param grid:
    :return:
    """
    distance_matrix = cdist(subgrid, grid)
    upper_tri_indices = np.triu_indices(distance_matrix.shape[0], k=1)
    indices = np.argwhere(distance_matrix[upper_tri_indices] <= tol)[0]
    return indices


def weighted_kmeans(grid_points: np.ndarray, f_weight: np.ndarray, centroids: np.ndarray, n_iter=200, centroid_tol=1.e-6,
                    safe_mode=False, verbose=True) -> np.ndarray:
    """

    TODOs:
     * Describe algorithm
     * Add latex
     * Try version of routine with MPI
     * Try version of routine with JIT
     * Try version of routine with cupy and numba

    :param grid_points: Real space grid
    :param f_weight: Weight function
    :param centroids: Initial centroids. A good choice is a set of N randomly (or uniformly) distributed
    points.
    The size of this array defines the number of interpolating grid points, N.
    These points must be part of the set of grid_points (?)
    :param n_iter: Number of iterations to find optimal centroids
    :return: interpolation_points: Grid points for interpolating vectors, as defined by optimised centroids
    """
    if safe_mode:
        indices = is_subgrid_on_grid(centroids, grid_points, 1.e-6)
        n_off_grid = len(indices)
        if n_off_grid> 0:
            print(f'{n_off_grid} out of {centroids.shape[0]} centroids are not defined on the real-space grid')
            print("# Index     Point")
            for i in indices:
                print(i, centroids[i, :])
            raise ValueError()

    N_r, dim = grid_points.shape

    if f_weight.shape[0] == N_r:
        err_msg = ("Number of sampling points defining the weight function differs to the size of the grid\n. "
                   "Weight function must be defined on the same real-space grid as grid_points")
        raise ValueError(err_msg)

    if verbose: print("Centroid Optimisation")
    updated_centroids = np.empty_like(centroids)

    for t in range(0, n_iter):
        clusters = assign_points_to_centroids(grid_points, centroids)
        updated_centroids = update_centroids(grid_points, f_weight, clusters)
        if verbose: print(f"Step {t}")
        converged = points_are_converged(centroids, updated_centroids, centroid_tol, verbose=verbose)
        if converged:
            return updated_centroids
        centroids = updated_centroids

    # Should return number of iterations
    return centroids
