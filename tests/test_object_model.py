import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pytest import approx
from app.object_model import ObjectModel


class TestObjectModel:
    def setup_method(self):
        self.obj = ObjectModel()

    def test_initial_state(self):
        assert np.allclose(self.obj.position, [0.0, 0.0, -3.0])
        assert self.obj.rotation_x == 0.0
        assert self.obj.rotation_y == 0.0
        assert self.obj.scale == 1.0

    def test_reset(self):
        self.obj.rotate_by(30, 45)
        self.obj.set_scale(2.0)
        self.obj.update(0.5)
        self.obj.reset()
        assert np.allclose(self.obj.position, [0.0, 0.0, -3.0])
        assert self.obj.rotation_x == 0.0
        assert self.obj.scale == 1.0

    def test_rotate_by(self):
        self.obj.rotate_by(10.0, 20.0, 5.0)
        self.obj.update(1.0)
        assert self.obj.rotation_x == approx(10.0, abs=0.01)
        assert self.obj.rotation_y == approx(20.0, abs=0.01)
        assert self.obj.rotation_z == approx(5.0, abs=0.01)

    def test_scale_by(self):
        self.obj.scale_by(0.5)
        self.obj.update(1.0)
        assert self.obj.scale == approx(1.5, abs=0.01)

    def test_scale_clamping(self):
        self.obj.set_scale(10.0)
        self.obj.update(1.0)
        assert self.obj.scale == approx(5.0, abs=0.01)
        self.obj.set_scale(0.01)
        self.obj.update(1.0)
        assert self.obj.scale == approx(0.1, abs=0.01)

    def test_model_matrix_shape(self):
        mat = self.obj.model_matrix
        assert mat.shape == (4, 4)

    def test_model_matrix_orthogonal(self):
        mat = self.obj.model_matrix
        assert np.allclose(mat[3], [0, 0, 0, 1])

    def test_translate_by(self):
        self.obj.translate_by(1.0, 2.0, -1.0)
        assert np.allclose(self.obj.position, [1.0, 2.0, -4.0])

    def test_get_state(self):
        state = self.obj.get_state()
        assert np.allclose(state.position, [0.0, 0.0, -3.0])
        assert np.allclose(state.rotation, [0.0, 0.0, 0.0])
        assert state.scale == 1.0
