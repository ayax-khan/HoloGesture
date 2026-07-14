import logging
import os
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ObjModel:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray):
        self.vertices = vertices
        self.indices = indices


def load_obj(filepath: str) -> Optional[ObjModel]:
    if not os.path.exists(filepath):
        logger.warning("OBJ file not found: %s", filepath)
        return None

    positions = []
    normals = []
    faces: List[Tuple[int, int, int]] = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("o ") or line.startswith("s ") or line.startswith("usemtl") or line.startswith("mtllib"):
                    continue
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "v":
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "vn":
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    verts = []
                    for token in parts[1:]:
                        v = token.split("/")
                        vi = int(v[0]) - 1
                        ni = int(v[2]) - 1 if len(v) > 2 and v[2] else 0
                        verts.append((vi, ni))
                    for i in range(1, len(verts) - 1):
                        faces.append((verts[0][0], verts[0][1],
                                      verts[i][0], verts[i][1],
                                      verts[i + 1][0], verts[i + 1][1]))
    except Exception as e:
        logger.error("Failed to parse OBJ %s: %s", filepath, e)
        return None

    if not positions or not faces:
        logger.warning("OBJ %s has no geometry", filepath)
        return None

    if not normals:
        normals = _compute_normals(positions, faces)

    verts_out = []
    idx_out = []
    unique: dict = {}
    for f in faces:
        for j in range(3):
            vi = f[j * 2]
            ni = f[j * 2 + 1]
            key = (vi, ni)
            if key not in unique:
                unique[key] = len(verts_out) // 6
                p = positions[vi] if vi < len(positions) else [0, 0, 0]
                n = normals[ni] if ni < len(normals) else [0, 1, 0]
                n_len = np.linalg.norm(n)
                if n_len > 0:
                    n = [n[0] / n_len, n[1] / n_len, n[2] / n_len]
                verts_out.extend([p[0], p[1], p[2], n[0], n[1], n[2]])
            idx_out.append(unique[key])

    vertices = np.array(verts_out, dtype=np.float32)
    indices = np.array(idx_out, dtype=np.uint32)
    logger.info("Loaded OBJ %s: %d verts, %d indices", filepath, len(vertices) // 6, len(indices))
    return ObjModel(vertices, indices)


def _compute_normals(positions: List[List[float]], faces: List) -> List[List[float]]:
    normals = [[0.0, 0.0, 0.0] for _ in positions]
    for f in faces:
        p0 = np.array(positions[f[0]])
        p1 = np.array(positions[f[2]])
        p2 = np.array(positions[f[4]])
        edge1 = p1 - p0
        edge2 = p2 - p0
        n = np.cross(edge1, edge2)
        n_len = np.linalg.norm(n)
        if n_len > 0:
            n = n / n_len
            for j in range(3):
                vi = f[j * 2]
                normals[vi][0] += n[0]
                normals[vi][1] += n[1]
                normals[vi][2] += n[2]
    for n in normals:
        n_len = np.linalg.norm(n)
        if n_len > 0:
            n[0] /= n_len
            n[1] /= n_len
            n[2] /= n_len
        else:
            n[2] = 1.0
    return normals
