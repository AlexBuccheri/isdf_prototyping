import numpy as np


from src.isdf_prototyping.interpolation_points import is_subgrid_on_grid


def test_is_subgrid_on_grid():
    # 2D grid, using arbitrary limits
    x_values = np.linspace(0, 10, 5)
    y_values = np.linspace(0, 1, 5)
    x, y = np.meshgrid(x_values, y_values)
    grid = np.vstack([x.ravel(), y.ravel()]).T
    assert grid.shape == (25, 2), "N grid points by M dimensions"

    subgrid = grid[2:4, :]
    indices = is_subgrid_on_grid(subgrid, grid, tol=1.e-6)
    assert not indices, "All subgrid points should be defined on the grid"

    # All elements
    noisy_subgrid = subgrid + 1.e-5
    n_sub = noisy_subgrid.shape[0]
    indices = is_subgrid_on_grid(noisy_subgrid, grid, tol=1.e-6)
    assert indices == [i for i in range(n_sub)], "No subgrid points should be defined on the grid"

    # Single element
    noisy_subgrid = np.copy(subgrid)
    noisy_subgrid[1, :] += 1.e-5
    indices = is_subgrid_on_grid(noisy_subgrid, grid, tol=1.e-6)
    assert indices == [1], "One subgrid point should not be on the grid"


def test_points_are_converged():
    pass


def test_assign_points_to_centroids():
    # Define a grid
    # Define a few centroids
    # Ensure one can do this by inspection
    # Confirm grid points are assigned to correct clusters

    # Do a second case where the points (choose several) are
    # equivdistant from two centroids, and confirm which clusters
    # each point gets assigned to - want this to be random, but will
    # need to build that mechanism in (then monkey patch for testing)
    pass


def test_update_centroids():
    # Can test with some trivial and specific weight functions
    pass


def test_weighted_kmeans():
    # Mock up the Gaussian example
    # Maybe do this test in Jupyter, for visualisation
    pass
