from unittest.mock import patch

import numpy as np

from src.isdf_prototyping.interpolation_points import (
    assign_points_to_centroids,
    is_subgrid_on_grid,
)


def construct_2d_grid(x_p, y_p) -> np.ndarray:
    """

    x and y args are the same ordering as with np.linspace
    i.e. start, stop, num

    :param x_p:
    :param y_p:
    :return:
    """
    x_values = np.linspace(*x_p)
    y_values = np.linspace(*y_p)
    x, y = np.meshgrid(x_values, y_values)
    grid = np.vstack([x.ravel(), y.ravel()]).T
    return grid


def test_is_subgrid_on_grid():
    # 2D grid, using arbitrary limits
    grid = construct_2d_grid((0, 10, 5), (0, 1, 5))
    assert grid.shape == (25, 2), "N grid points by M dimensions"

    subgrid = grid[2:4, :]
    indices = is_subgrid_on_grid(subgrid, grid, tol=1.0e-6)
    assert not indices, "All subgrid points should be defined on the grid"

    # All elements
    noisy_subgrid = subgrid + 1.0e-5
    n_sub = noisy_subgrid.shape[0]
    indices = is_subgrid_on_grid(noisy_subgrid, grid, tol=1.0e-6)
    assert indices == [i for i in range(n_sub)], "No subgrid points should be defined on the grid"

    # Single element
    noisy_subgrid = np.copy(subgrid)
    noisy_subgrid[1, :] += 1.0e-5
    indices = is_subgrid_on_grid(noisy_subgrid, grid, tol=1.0e-6)
    assert indices == [1], "One subgrid point should not be on the grid"


def test_assign_points_to_centroids():
    """
    Define a grid comprised of circular grid points
    Place one centroid at its centre and the other away from it
    All points should assign to the same centroid
    """
    centroids = np.array([[0, 0], [5, 0]])
    radius = 2
    n_points = 10
    grid = np.empty(shape=(n_points, 2))
    for i in range(0, n_points):
        theta = (float(i + 1) / float(n_points)) * 2 * np.pi
        grid[i, :] = centroids[0, :] + radius * np.array([np.cos(theta), np.sin(theta)])

    clusters = assign_points_to_centroids(grid, centroids)
    assert clusters[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "All grid points in first cluster"
    assert clusters[1] == [], "No grid points in second cluster"


def test_assign_points_equidistant_to_centroids():
    """Test assignment of points that are equidistant to two centroids"""
    # Three points all equidistant between two centroids
    grid = np.array([[0, 0], [0.5, 0.5], [1.0, 1.0]])
    centroids = np.array([[0.0, 1.0], [1.0, 0]])

    clusters = assign_points_to_centroids(grid, centroids)
    assert set(clusters[0]) | set(clusters[1]) == {0, 1, 2}, \
        "All points assigned randomly between the two clusters "

    with patch("isdf_prototyping.interpolation_points.np.random.choice") as mock_random:
        mock_random.return_value = 0
        clusters = assign_points_to_centroids(grid, centroids)
        assert clusters == [[0, 1, 2], []], "Mock random so all equi-d points go to cluster 0"

        mock_random.return_value = 1
        clusters = assign_points_to_centroids(grid, centroids)
        assert clusters == [[], [0, 1, 2]], "Mock random so all equi-d points go to cluster 1"


def test_update_centroids():
    grid = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                     [2, 0], [3, 0], [2, 1], [3, 1]])

    assert grid.shape == (8, 2), 'Npoints x dimensions'
    centroids = np.array([[0.5, 0.5], [2.5, 0.5]])

    expected_clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
    clusters = assign_points_to_centroids(grid, centroids)
    assert clusters == expected_clusters


def test_points_are_converged():
    pass


# def test_weighted_kmeans():
#     # Mock up the Gaussian example
#     # Maybe do this test in Jupyter, for visualisation
#
#     # Example of how to plot the overlying Voronoi tessellations
#     from scipy.spatial import Voronoi, voronoi_plot_2d
#     import matplotlib.pyplot as plt
#
#     # Generate random 2D points
#     points = np.random.rand(10, 2)
#     vor = Voronoi(points)
#
#     # Plot Voronoi diagram
#     fig, ax = plt.subplots(figsize=(8, 6))
#     voronoi_plot_2d(vor, ax=ax, show_vertices=False)
#
#     # Plot input points
#     ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
#
#     # Customize plot appearance
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('Voronoi Tessellations')
#     ax.grid(True)
#     plt.show()
