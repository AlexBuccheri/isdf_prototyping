import io
import sys
from unittest.mock import patch

import numpy as np

from src.isdf_prototyping.interpolation_points import (
    assign_points_to_centroids,
    is_subgrid_on_grid, update_centroids, points_are_converged, weighted_kmeans,
)


def construct_2d_grid(x_p, y_p) -> np.ndarray:
    """

    x and y args are the same ordering as with np.linspace
    i.e. start, stop, num

    * JIT:
    Can't use numba as np.meshgrid is not supported

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

    N_r = 8
    assert grid.shape == (N_r, 2), 'Npoints x dimensions'
    centroids = np.array([[0.5, 0.5], [2.5, 0.5]])

    expected_clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
    clusters = assign_points_to_centroids(grid, centroids)
    assert clusters == expected_clusters

    # Centroids are chosen in optimal positions.
    # With uniform weighting, one does not expect them to change
    # Note that differences in centroid positions has no effect on the call
    # - what matters is what cluster grid points are assigned to
    uniform_weight = np.empty(shape=N_r)
    uniform_weight.fill(1. / N_r)
    new_centroids = update_centroids(grid, uniform_weight, expected_clusters)
    assert np.allclose(new_centroids, centroids)


class StdoutCapture:
    """ Custom context manager (defines __enter__ and __exit__ methods for the with block).
        Therefore, need to exit the with block before the captured stdout is returned

    Could have also used @contextmanager to write a functional version
    """

    def __enter__(self):
        """ Redirect sys.stdout to an StringIO on entry
        """
        self.stdout_buffer = io.StringIO()
        sys.stdout = self.stdout_buffer
        return self.stdout_buffer

    def __exit__(self, exc_type, exc_value, traceback):
        """ Reset IO stream on exit
        """
        sys.stdout = sys.__stdout__
        if exc_type is not None:
            # If an exception occurred, let it propagate
            return False


def test_points_are_converged():
    # 2D grid, using arbitrary limits
    grid = construct_2d_grid((0, 10, 5), (0, 1, 5))
    assert grid.shape == (25, 2), "N grid points by M dimensions"

    # Add an erroneous point
    updated_grid = np.copy(grid)
    updated_grid[0, :] += np.array([0.1, 0.1])

    converged = points_are_converged(updated_grid, grid, tol=1.e-6)
    assert not converged, "First point differs between grids"

    # Test verbose
    with StdoutCapture() as stdout_buffer:
        # Multiple sources of failure
        updated_grid[1:4, :] += np.array([0.1, 0.1])
        converged = points_are_converged(updated_grid, grid, tol=1.e-6, verbose=True)
        captured_stdout = stdout_buffer

    assert repr(captured_stdout.getvalue()) == ("'Convergence: 4 points out of 25 are not converged:\\n"
                                                "# Current Point    Prior Point    |ri - r_{i-1}|   tol\\n"
                                                "[0.1 0.1] [0. 0.] 0.14142135623730953 1e-06\\n"
                                                "[2.6 0.1] [2.5 0. ] 0.14142135623730956 1e-06\\n"
                                                "[5.1 0.1] [5. 0.] 0.14142135623730925 1e-06\\n"
                                                "[7.6 0.1] [7.5 0. ] 0.14142135623730925 1e-06\\n'")


def test_weighted_kmeans():
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # Place a Gaussian distribution in each quadrant of the cubic grid
    mu1 = [2.5, 2.5]
    mu2 = [7.5, 2.5]
    mu3 = [2.5, 7.5]
    mu4 = [7.5, 7.5]
    cov = [[0.75, 0], [0, 0.75]]  # Covariance matrix: Diagonals are variances == std^2 i.e. (Gaussian width)^2

    # Create a grid of points in 2D space
    x = np.linspace(0, 10, 40)
    y = np.linspace(0, 10, 40)
    X, Y = np.meshgrid(x, y)
    grid = np.dstack((X, Y))
    # grid[1, 0, :] == grid2[40, :]

    z1 = multivariate_normal.pdf(grid, mean=mu1, cov=cov)
    z2 = multivariate_normal.pdf(grid, mean=mu2, cov=cov)
    z3 = multivariate_normal.pdf(grid, mean=mu3, cov=cov)
    z4 = multivariate_normal.pdf(grid, mean=mu4, cov=cov)
    z_total = z1 + z2 + z3 + z4

    # Plot the 2D Gaussian PDF
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, z1, cmap='viridis')
    # plt.show()

    # Number of interpolation points
    grid_reshaped = grid.reshape(-1, 2)
    assert grid_reshaped.shape == (1600, 2)
    n_inter = 48

    # random_indices = np.sort(np.random.choice(grid_reshaped.shape[0], size=n_inter))
    random_indices =  [10, 19, 31, 39, 93, 119, 148, 242, 255, 262, 267, 359, 364, 377,
    411,  436,  440,  457,  468,  477,  525,  562,  565,  592,  614,  624,  628,  662,
    754,  832,  855,  877,  894,  971,  985, 1028, 1086, 1208, 1209, 1261, 1269, 1344,
   1425, 1461, 1493, 1498, 1582, 1590]
    assert len(random_indices) == n_inter

    initial_centroids = grid_reshaped[random_indices, :]

    # plt.figure(figsize=(8, 6))
    # plt.contourf(X, Y, z_total, cmap='viridis')
    # plt.colorbar(label='Probability Density')
    # plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], color='red', label='Initial Centroids')
    # plt.show()

    weights = z_total.reshape(-1)
    assert weights.shape == (1600,)
    centroids, iter = weighted_kmeans(grid_reshaped, weights, initial_centroids,
                                      n_iter=100, centroid_tol=1.0e-9,
                                      safe_mode=True, verbose=False)


    print(iter)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, z_total, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Initial Centroids')
    plt.show()


    # # Example of how to plot the overlying Voronoi tessellations
    # from scipy.spatial import Voronoi, voronoi_plot_2d
    # import matplotlib.pyplot as plt
    #
    # # Generate random 2D points
    # points = np.random.rand(10, 2)
    # vor = Voronoi(points)
    #
    # # Plot Voronoi diagram
    # fig, ax = plt.subplots(figsize=(8, 6))
    # voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    #
    # # Plot input points
    # ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
    #
    # # Customize plot appearance
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Voronoi Tessellations')
    # ax.grid(True)
    # plt.show()
