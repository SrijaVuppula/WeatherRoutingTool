import numpy as np
from geographiclib.geodesic import Geodesic

from WeatherRoutingTool.algorithms.data_utils import distance, time_diffs


def test_distance_returns_accumulated_geodesic_distance():
    route = np.array([
        [54.9, 13.2],
        [54.7, 13.4],
        [54.5, 13.7],
    ])

    geod = Geodesic.WGS84
    d01 = geod.Inverse(route[0, 0], route[0, 1], route[1, 0], route[1, 1])["s12"]
    d12 = geod.Inverse(route[1, 0], route[1, 1], route[2, 0], route[2, 1])["s12"]

    dists = distance(route)
    expected = np.array([0.0, d01, d01 + d12])

    assert np.allclose(dists, expected)


def test_distance_is_non_decreasing():
    route = np.array([
        [54.9, 13.2],
        [54.7, 13.4],
        [54.5, 13.7],
        [54.2, 13.9],
    ])

    dists = distance(route)

    assert np.all(np.diff(dists) >= 0)


def test_time_diffs_matches_distance_divided_by_speed():
    route = np.array([
        [54.9, 13.2],
        [54.7, 13.4],
        [54.5, 13.7],
    ])
    speed = 6.0

    diffs = time_diffs(speed, route)

    assert np.allclose(diffs, distance(route) / speed)
