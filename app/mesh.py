from enum import Enum, auto
from typing import Tuple

import numpy as np
from OpenGL import GL


class ObjectType(Enum):
    SOLID_CUBE = auto()
    SOLID_SPHERE = auto()
    WIREFRAME_CUBE = auto()
    HOLOGRAPHIC_SPHERE = auto()


OBJECT_NAMES = {
    ObjectType.SOLID_CUBE: "Solid Cube",
    ObjectType.SOLID_SPHERE: "Solid Sphere",
    ObjectType.WIREFRAME_CUBE: "Wireframe Cube",
    ObjectType.HOLOGRAPHIC_SPHERE: "HoloSphere",
}


class Mesh:
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(2)
        self.index_count = 0
        self.primitive_type = GL.GL_TRIANGLES

    def build(self, obj_type: ObjectType, detail: int = 32):
        if obj_type == ObjectType.SOLID_CUBE:
            vertices, indices = self._cube_data()
            self.primitive_type = GL.GL_TRIANGLES
        elif obj_type == ObjectType.WIREFRAME_CUBE:
            vertices, indices = self._wireframe_cube_data()
            self.primitive_type = GL.GL_LINES
        elif obj_type == ObjectType.SOLID_SPHERE:
            vertices, indices = self._sphere_data(detail, detail // 2, solid=True)
            self.primitive_type = GL.GL_TRIANGLES
        elif obj_type == ObjectType.HOLOGRAPHIC_SPHERE:
            vertices, indices = self._sphere_data(detail, detail // 2, solid=False)
            self.primitive_type = GL.GL_TRIANGLES
        else:
            vertices, indices = self._cube_data()
            self.primitive_type = GL.GL_TRIANGLES

        self.index_count = len(indices)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vbo[1])
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        stride = 24
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

    def _cube_data(self):
        verts = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        faces = [
            (0, 1, 3, 2), (4, 6, 7, 5),
            (0, 4, 5, 1), (2, 3, 7, 6),
            (0, 2, 6, 4), (1, 5, 7, 3)
        ]
        normals = [
            (0, 0, -1), (0, 0, 1),
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0)
        ]
        vertices = []
        indices = []
        idx = 0
        for face, normal in zip(faces, normals):
            for vi in face:
                v = verts[vi]
                vertices.extend([v[0], v[1], v[2], normal[0], normal[1], normal[2]])
            base = idx
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
            idx += 4
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def _wireframe_cube_data(self):
        verts = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        vertices = []
        for v in verts:
            vertices.extend([v[0], v[1], v[2], 0.0, 0.0, 1.0])
        indices = []
        for e in edges:
            indices.extend([e[0], e[1]])
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def _sphere_data(self, slices: int, stacks: int, solid: bool = True):
        vertices = []
        indices = []
        for i in range(stacks + 1):
            phi = np.pi * i / stacks
            for j in range(slices + 1):
                theta = 2 * np.pi * j / slices
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
                vertices.extend([x, y, z, x, y, z])
        for i in range(stacks):
            for j in range(slices):
                first = i * (slices + 1) + j
                second = first + slices + 1
                if solid:
                    indices.extend([first, second, first + 1])
                    indices.extend([second, second + 1, first + 1])
                else:
                    indices.extend([first, second, first + 1, second, second + 1, first + 1])
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def delete(self):
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(2, self.vbo)
