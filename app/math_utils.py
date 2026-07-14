import numpy as np
from typing import Tuple


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vector(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def low_pass_filter(current: float, previous: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous


def low_pass_filter_vector(current: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * current + (1.0 - alpha) * previous


def rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def translation_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def scaling_matrix(s: float) -> np.ndarray:
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, s, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def landmark_to_numpy(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
