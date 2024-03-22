from isdf_prototyping.interpolation_points import (
    assign_points_to_centroids,
    is_subgrid_on_grid,
    points_are_converged,
)


def test_is_subgrid_on_grid():
    # Mock up subgrid that is calved off from a square grid
    # Confirm it's ok
    # Add some noise to a few points - confirm it gets caught
    pass


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
