import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.math_utils import (
    euclidean_distance, angle_between_vectors, lerp, lerp_vector,
    low_pass_filter, rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    translation_matrix, scaling_matrix, perspective_matrix
)


class TestMathUtils:
    def test_euclidean_distance_zero(self):
        assert euclidean_distance(np.zeros(3), np.zeros(3)) == 0.0

    def test_euclidean_distance_positive(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        assert euclidean_distance(a, b) == 3.0

    def test_angle_between_vectors_parallel(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        assert angle_between_vectors(a, b) == 0.0

    def test_angle_between_vectors_perpendicular(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(angle_between_vectors(a, b) - 90.0) < 0.001

    def test_lerp(self):
        assert lerp(0.0, 10.0, 0.5) == 5.0
        assert lerp(0.0, 10.0, 0.0) == 0.0
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_lerp_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 20.0])
        result = lerp_vector(a, b, 0.5)
        assert np.allclose(result, [5.0, 10.0])

    def test_low_pass_filter(self):
        result = low_pass_filter(1.0, 0.0, 0.6)
        assert abs(result - 0.6) < 0.001

    def test_rotation_matrix_x(self):
        R = rotation_matrix_x(90.0)
        assert np.allclose(R[:3, :3].dot([0, 1, 0]), [0, 0, 1], atol=1e-6)

    def test_rotation_matrix_y(self):
        R = rotation_matrix_y(90.0)
        assert np.allclose(R[:3, :3].dot([1, 0, 0]), [0, 0, -1], atol=1e-6)

    def test_translation_matrix(self):
        T = translation_matrix(1, 2, 3)
        assert np.allclose(T[:3, 3], [1, 2, 3])

    def test_scaling_matrix(self):
        S = scaling_matrix(2.0)
        assert np.allclose(S[:3, :3].dot([1, 1, 1]), [2, 2, 2])

    def test_perspective_matrix_properties(self):
        P = perspective_matrix(45.0, 1.333, 0.1, 100.0)
        assert P.shape == (4, 4)

    def test_zero_norm_angle(self):
        angle = angle_between_vectors(np.zeros(3), np.ones(3))
        assert angle == 0.0
