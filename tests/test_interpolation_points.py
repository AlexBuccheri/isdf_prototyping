import io
import sys
from unittest.mock import patch

import numpy as np
from scipy.stats import multivariate_normal

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
    """Test assignment of points that are equidistant to two initial_centroids"""
    # Three points all equidistant between two initial_centroids
    grid = np.array([[0, 0], [0.5, 0.5], [1.0, 1.0]])
    centroids = np.array([[0.0, 1.0], [1.0, 0]])

    clusters = assign_points_to_centroids(grid, centroids)
    assert set(clusters[0]) | set(clusters[1]) == {0, 1, 2}, \
        "All points assigned randomly between the two clusters "

    # TODO(Alex) reintroduce once testing of k-means is done
    # with patch("isdf_prototyping.interpolation_points.np.random.choice") as mock_random:
    #     mock_random.return_value = 0
    #     clusters = assign_points_to_centroids(grid, initial_centroids)
    #     assert clusters == [[0, 1, 2], []], "Mock random so all equi-d points go to cluster 0"
    #
    #     mock_random.return_value = 1
    #     clusters = assign_points_to_centroids(grid, initial_centroids)
    #     assert clusters == [[], [0, 1, 2]], "Mock random so all equi-d points go to cluster 1"


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
    assert grid.shape == (40, 40, 2)

    z1 = multivariate_normal.pdf(grid, mean=mu1, cov=cov)
    z2 = multivariate_normal.pdf(grid, mean=mu2, cov=cov)
    z3 = multivariate_normal.pdf(grid, mean=mu3, cov=cov)
    z4 = multivariate_normal.pdf(grid, mean=mu4, cov=cov)
    z_total = z1 + z2 + z3 + z4

    # Reshape Gaussians into required shape for weights
    weights = z_total.reshape(-1)
    assert weights.shape == (1600,)

    # Reshape grid into required shape
    grid_reshaped = grid.reshape(-1, 2)
    assert grid_reshaped.shape == (1600, 2)

    # Number of interpolation points
    n_inter = 48

    # random_indices = np.sort(np.random.choice(grid_reshaped.shape[0], size=n_inter))
    random_indices = [10, 19, 31, 39, 93, 119, 148, 242, 255, 262, 267, 359, 364, 377,
    411,  436,  440,  457,  468,  477,  525,  562,  565,  592,  614,  624,  628,  662,
    754,  832,  855,  877,  894,  971,  985, 1028, 1086, 1208, 1209, 1261, 1269, 1344,
   1425, 1461, 1493, 1498, 1582, 1590]
    assert len(random_indices) == n_inter

    initial_centroids = grid_reshaped[random_indices, :]

    # Perform a single iteration and confirm the initial_centroids returned never change
    centroids, iter = weighted_kmeans(grid_reshaped, weights, initial_centroids,
                                      n_iter=1, centroid_tol=1.0e-9,
                                      safe_mode=True, verbose=False)
    ref_one_iter_centroids = np.array([[2.11114651, 0.80404266],
 [4.93651777, 0.44107855],
 [8.19662874, 0.67798632],
 [9.20652102, 0.17751151],
 [2.91578925, 1.03767698],
 [9.20652102, 1.05475048],
 [7.54277176, 1.0631469 ],
 [1.0172123 , 1.3270902 ],
 [3.45077672, 1.68495259],
 [5.87134816, 1.88474071],
 [7.16714586, 1.78691257],
 [9.59164901, 1.86711955],
 [1.49866405, 2.13017755],
 [4.09122977, 2.31948319],
 [2.630152  , 2.50787456],
 [8.58077489, 2.22300062],
 [0.2563042 , 2.73427841],
 [4.13971911, 2.98993416],
 [7.25120505, 2.70679555],
 [9.30486898, 3.15684304],
 [1.6457442 , 3.08626129],
 [0.58281683, 3.67890713],
 [1.84449649, 3.90193938],
 [8.16847866, 3.35373193],
 [3.23532173, 3.71726697],
 [6.25200669, 3.54180483],
 [7.2735667 , 3.90764901],
 [5.52411858, 4.03830951],
 [8.77299323, 4.28327255],
 [8.06927001, 5.60320802],
 [4.05692773, 5.49405499],
 [9.14162658, 6.61357341],
 [3.70724023, 6.19123547],
 [2.86222864, 6.54355945],
 [6.3398635 , 6.49374389],
 [7.5032693 , 6.6967575 ],
 [1.48110757, 6.89845869],
 [1.70739303, 8.04467417],
 [2.80239068, 7.75954148],
 [4.98234286, 7.6007253 ],
 [7.78975606, 7.83130057],
 [6.38502406, 8.05012528],
 [6.80663179, 8.99082984],
 [5.51472405, 9.00446659],
 [3.03660096, 8.89152615],
 [4.26201438, 8.82678713],
 [6.13773648, 9.8847534 ],
 [8.08448695, 9.25588087]])

    assert np.allclose(centroids, ref_one_iter_centroids), 'Centroids always returned same after 1 iter'

    # With the random assign at equivdistance removed,
    centroids, iter = weighted_kmeans(grid_reshaped, weights, initial_centroids,
                                      n_iter=100, centroid_tol=1.0e-9,
                                      safe_mode=True, verbose=False)

    assert iter == 18, "For a fixed seed of centroids, the algorithm converges to the same interpolation points each time."

    ref_centroids = np.array([[2.23203643, 1.47684664],
 [5.5316338 , 1.9741898 ],
 [7.83555543, 1.27123827],
 [8.51564229, 0.67320234],
 [3.04794118, 0.90118687],
 [8.70862954, 1.63798743],
 [6.84592092, 0.94636725],
 [1.20715416, 1.29624651],
 [3.23574429, 1.82086744],
 [6.46358304, 1.96575468],
 [7.30199585, 1.85627307],
 [9.31698114, 2.18012182],
 [1.84584149, 2.22207151],
 [3.99862222, 2.26133899],
 [2.73554832, 2.47058413],
 [8.17621923, 2.23919523],
 [0.97575504, 2.59601138],
 [3.67008745, 3.26251379],
 [7.32598254, 2.62523899],
 [8.9756401 , 3.01423676],
 [1.97546603, 3.0670512 ],
 [1.14195314, 3.76136972],
 [2.27101012, 4.05287169],
 [8.0626232 , 3.12736177],
 [2.85149722, 3.39538029],
 [6.44034167, 2.91449385],
 [7.24781495, 3.7018047 ],
 [6.16320945, 3.91624513],
 [8.35437481, 4.08376274],
 [7.81140662, 5.9978151 ],
 [3.26477232, 5.71957533],
 [8.70221595, 7.03796999],
 [3.45511503, 6.99187557],
 [2.44291226, 6.4936108 ],
 [6.50913356, 6.57421076],
 [7.52628572, 6.95605009],
 [1.48420575, 6.94426319],
 [1.51453492, 8.1614986 ],
 [2.55662442, 7.6409192 ],
 [4.82813681, 7.19923193],
 [7.85392558, 7.76711644],
 [6.80548628, 7.53686908],
 [7.18118555, 8.37471305],
 [6.05920862, 8.22246161],
 [2.53717063, 8.90743691],
 [3.5318929 , 8.17623078],
 [7.33043045, 9.24135951],
 [8.55645875, 8.4467815 ]])

    assert np.allclose(centroids, ref_centroids), "For a fixed seed of centroids, the algorithm converges to the same interpolation points each time."



