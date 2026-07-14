import numpy as np
from typing import Optional
from dataclasses import dataclass

from app.math_utils import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    translation_matrix, scaling_matrix, lerp, lerp_vector
)


@dataclass
class ObjectState:
    position: np.ndarray
    rotation: np.ndarray
    scale: float


class ObjectModel:
    MAX_SCALE = 5.0
    MIN_SCALE = 0.1
    MAX_TILT = 180.0
    MAX_TRANSLATE = 5.0

    def __init__(self):
        self.position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.scale = 1.0
        self.target_scale = 1.0
        self.target_rotation_x = 0.0
        self.target_rotation_y = 0.0
        self.target_rotation_z = 0.0
        self.velocity = np.zeros(3, dtype=np.float32)
        self.damping = 0.92

    def reset(self):
        self.position[:] = [0.0, 0.0, -3.0]
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.scale = 1.0
        self.target_scale = 1.0
        self.target_rotation_x = 0.0
        self.target_rotation_y = 0.0
        self.target_rotation_z = 0.0
        self.velocity[:] = 0.0

    def rotate_by(self, dx: float, dy: float, dz: float = 0.0):
        self.target_rotation_x = max(-self.MAX_TILT, min(self.MAX_TILT, self.target_rotation_x + dx))
        self.target_rotation_y = (self.target_rotation_y + dy) % 360.0
        self.target_rotation_z = max(-self.MAX_TILT, min(self.MAX_TILT, self.target_rotation_z + dz))

    def translate_by(self, dx: float, dy: float, dz: float = 0.0):
        new_pos = self.position + np.array([dx, dy, dz])
        self.position[0] = max(-self.MAX_TRANSLATE, min(self.MAX_TRANSLATE, new_pos[0]))
        self.position[1] = max(-self.MAX_TRANSLATE, min(self.MAX_TRANSLATE, new_pos[1]))
        self.position[2] = max(-self.MAX_TRANSLATE, min(self.MAX_TRANSLATE, new_pos[2]))

    def set_scale(self, s: float):
        self.target_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, s))

    def scale_by(self, delta: float):
        self.target_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self.target_scale + delta))

    def update(self, dt: float):
        smooth = 1.0 - (0.85 ** (dt * 60))
        self.rotation_x = lerp(self.rotation_x, self.target_rotation_x, smooth)
        self.rotation_y = lerp(self.rotation_y, self.target_rotation_y, smooth)
        self.rotation_z = lerp(self.rotation_z, self.target_rotation_z, smooth)
        self.scale = lerp(self.scale, self.target_scale, smooth)
        self.velocity *= self.damping
        if np.linalg.norm(self.velocity) > 0.001:
            self.position += self.velocity * dt
        else:
            self.velocity[:] = 0.0

    @property
    def model_matrix(self) -> np.ndarray:
        T = translation_matrix(self.position[0], self.position[1], self.position[2])
        Rx = rotation_matrix_x(self.rotation_x)
        Ry = rotation_matrix_y(self.rotation_y)
        Rz = rotation_matrix_z(self.rotation_z)
        S = scaling_matrix(self.scale)
        return T @ Ry @ Rx @ Rz @ S

    def get_state(self) -> ObjectState:
        return ObjectState(
            position=self.position.copy(),
            rotation=np.array([self.rotation_x, self.rotation_y, self.rotation_z]),
            scale=self.scale
        )
