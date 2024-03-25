""" Module for functions that find interpolation grid points:

* QR decomposition with column pivoting
* Weighted k-means clustering
"""

from typing import List

import numpy as np
from scipy.spatial.distance import cdist

cluster_type = List[List[int]]


def is_subgrid_on_grid(subgrid: np.ndarray, grid: np.ndarray, tol: float) -> list:
    """Return indices of subgrid points not present in grid.

    Parallelisation:
    * Won't benefit from Numba, as only called once
    * Loop shouldn't be large enough to parallelise with MPI
      - One could thread it
    * Function might benefit from cupy

    :param subgrid:
    :param grid:
    :return:
    """
    N_sub = subgrid.shape[0]
    indices = []
    for i in range(N_sub):
        centred_on_point_i = np.linalg.norm(grid - subgrid[i], axis=1)
        matched_indices = np.argwhere(centred_on_point_i <= tol)
        if matched_indices.size == 0:
            indices.append(i)
    return indices


def assign_points_to_centroids(grid_points: np.ndarray, centroids: np.ndarray) -> cluster_type:
    """Assign each grid point to the closest centroid. A centroid and its set of nearest
    grid points defines a cluster.

    Refactor:
    * Could work on the distance matrix for all points
    * Could simplify such that I just use argmin. Not likely that points will be equidistant
      in all but symmetric, toy problems

    Parallelisation Strategies:
    * Could benefit from MPI distribution of loop
    * ~Could benefit from NUMBA, as called multiple times~   cdist is scipy, not numpy
      - Would have to @overload(scipy.spatial.distance.cdist) and supply my own compiled implementation
    * Could benefit from cupy as *some* numpy operations are used

    :param grid_points: Grid point
    :param centroids: Centroid points

    :return: clusters: List of length N interpolation points. Each element corresponds to a cluster,
    which indexes all grid points associated with it
    """
    N_r, dim = grid_points.shape
    N_interp = centroids.shape[0]
    clusters: cluster_type = [[] for _ in range(N_interp)]

    # Computing distance matrix per grid point because I want to distribute this loop
    # (keeping a fortran implementation in mind)
    for ir in range(0, N_r):
        # Need to retain a 2D array for cdist
        point = grid_points[ir, :][None, :]
        # distance matrix has shape (1, N_interp)
        distance_matrix = cdist(point, centroids).reshape(-1)
        # min(|r_{ir} - centroids|)
        min_index = np.argmin(distance_matrix)
        # If two or more elements are equally minimal, argmin will always return the first instance
        # Instead, we find all equally minimum indices
        min_indices = np.argwhere(distance_matrix == distance_matrix[min_index])[:, 0]
        icen = np.random.choice(min_indices)
        clusters[icen].append(ir)
    return clusters


def update_centroids(grid_points, f_weight, clusters: cluster_type) -> np.ndarray:
    """Compute a new set of centroids

    We have as many clusters as we do centroids

    :param grid_points: Grid
    :param f_weight: Weight function
    :param clusters: Grid point indices associated with each cluster
    :return: updated_centroids: Updated centroids
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
    """Given the difference in two sets of points, determine whether the updated
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
            print("Convergence: All points converged")
        else:
            print(f"Convergence: {len(indices)} points out of {N} are not converged:")
            print("# Current Point    Prior Point    |ri - r_{i-1}|   tol")
            for i in indices:
                print(updated_points[i, :], points[i, :], norm[i], tol)

    return converged


def verbose_print(*args, verbose=True, **kwargs):
    if verbose:
        print(*args, **kwargs)


def weighted_kmeans(
    grid_points: np.ndarray,
    f_weight: np.ndarray,
    centroids: np.ndarray,
    n_iter=200,
    centroid_tol=1.0e-6,
    safe_mode=False,
    verbose=True,
) -> np.ndarray:
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
        indices = is_subgrid_on_grid(centroids, grid_points, 1.0e-6)
        n_off_grid = len(indices)
        if n_off_grid > 0:
            print(
                f"{n_off_grid} out of {centroids.shape[0]} centroids are not defined on the real-space grid"
            )
            print("# Index     Point")
            for i in indices:
                print(i, centroids[i, :])
            raise ValueError()

    N_r, dim = grid_points.shape

    if f_weight.shape[0] == N_r:
        err_msg = (
            "Number of sampling points defining the weight function differs to the size of the grid\n. "
            "Weight function must be defined on the same real-space grid as grid_points"
        )
        raise ValueError(err_msg)

    verbose_print("Centroid Optimisation", verbose)

    for t in range(0, n_iter):
        clusters = assign_points_to_centroids(grid_points, centroids)
        updated_centroids = update_centroids(grid_points, f_weight, clusters)
        verbose_print(f"Step {t}", verbose)
        converged = points_are_converged(centroids, updated_centroids, centroid_tol, verbose=verbose)
        if converged:
            return updated_centroids, t
        centroids = updated_centroids

    return centroids, t
