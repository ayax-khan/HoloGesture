import logging
from typing import Dict, Optional, Tuple

import numpy as np
from OpenGL import GL

from config import config
from app.object_model import ObjectModel
from app.math_utils import perspective_matrix, look_at_matrix
from app.mesh import Mesh, ObjectType
from app.shaders import SHADER_SOURCES, VERTEX_SHADER_SRC
from app.errors import RendererError, ShaderError

logger = logging.getLogger(__name__)


class ShaderProgram:
    def __init__(self):
        self.program = GL.glCreateProgram()
        self._attached = []

    def attach_shader(self, source: str, shader_type: int):
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            GL.glDeleteShader(shader)
            raise ShaderError(f"Shader compile error: {log.decode()}")
        GL.glAttachShader(self.program, shader)
        self._attached.append(shader)

    def link(self):
        GL.glLinkProgram(self.program)
        if not GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(self.program)
            raise ShaderError(f"Program link error: {log.decode()}")
        for shader in self._attached:
            GL.glDetachShader(self.program, shader)
            GL.glDeleteShader(shader)
        self._attached.clear()

    def use(self):
        GL.glUseProgram(self.program)

    def uniform_location(self, name: str) -> int:
        return GL.glGetUniformLocation(self.program, name)

    def set_mat4(self, name: str, mat: np.ndarray):
        loc = self.uniform_location(name)
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mat.T.flatten())

    def set_vec3(self, name: str, vec: np.ndarray):
        loc = self.uniform_location(name)
        GL.glUniform3fv(loc, 1, vec)

    def set_float(self, name: str, val: float):
        loc = self.uniform_location(name)
        GL.glUniform1f(loc, val)

    def delete(self):
        GL.glDeleteProgram(self.program)


class Renderer:
    SHADER_KEYS = {"solid": "solid", "wireframe": "wireframe", "holographic": "holographic"}

    def __init__(self):
        self._shaders: Dict[str, ShaderProgram] = {}
        self._meshes: Dict[ObjectType, Mesh] = {}
        self._initialized = False
        self._time = 0.0
        self._view_matrix: Optional[np.ndarray] = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._current_obj_type = ObjectType.SOLID_CUBE
        self._camera_orbit_angle = 0.0
        self._camera_height = 1.0
        self._camera_distance = 3.5

    def initialize(self):
        logger.info("Initializing renderer")
        try:
            self._build_shaders()
            self._build_meshes()
            self._initialized = True
            logger.info("Renderer initialized successfully")
        except ShaderError as e:
            logger.error(f"Shader initialization failed: {e}")
            raise RendererError(f"Renderer init failed: {e}") from e

    def _build_shaders(self):
        for key, (vs, fs) in SHADER_SOURCES.items():
            prog = ShaderProgram()
            prog.attach_shader(vs, GL.GL_VERTEX_SHADER)
            prog.attach_shader(fs, GL.GL_FRAGMENT_SHADER)
            prog.link()
            self._shaders[key] = prog

    def _build_meshes(self):
        for obj_type in ObjectType:
            mesh = Mesh()
            mesh.build(obj_type, config.render.mesh_detail)
            self._meshes[obj_type] = mesh

    @property
    def current_object_type(self) -> ObjectType:
        return self._current_obj_type

    @current_object_type.setter
    def current_object_type(self, obj_type: ObjectType):
        if obj_type in self._meshes:
            self._current_obj_type = obj_type
            logger.info(f"Switched to {obj_type.name}")

    def resize(self, width: int, height: int):
        GL.glViewport(0, 0, width, height)
        aspect = width / max(height, 1)
        self._projection_matrix = perspective_matrix(45.0, aspect, 0.1, 100.0)

    def _update_view(self):
        cx = self._camera_distance * np.sin(np.radians(self._camera_orbit_angle))
        cz = self._camera_distance * np.cos(np.radians(self._camera_orbit_angle))
        eye = np.array([cx, self._camera_height, cz], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._view_matrix = look_at_matrix(eye, center, up)

    def orbit_camera(self, delta_angle: float):
        self._camera_orbit_angle = (self._camera_orbit_angle + delta_angle) % 360.0

    def zoom_camera(self, delta: float):
        self._camera_distance = max(1.5, min(10.0, self._camera_distance + delta))

    def render(self, obj: ObjectModel, dt: float):
        if not self._initialized:
            return

        self._time += dt
        self._update_view()

        GL.glClearColor(*config.render.clear_color)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        obj_type = self._current_obj_type
        if obj_type == ObjectType.WIREFRAME_CUBE:
            shader_key = "wireframe"
        elif obj_type == ObjectType.HOLOGRAPHIC_SPHERE:
            shader_key = "holographic"
        else:
            shader_key = "solid"

        prog = self._shaders[shader_key]
        mesh = self._meshes[obj_type]

        prog.use()
        prog.set_mat4("uProjection", self._projection_matrix)
        prog.set_mat4("uView", self._view_matrix)
        prog.set_mat4("uModel", obj.model_matrix)

        light_pos = np.array([2.0, 3.0, 2.0], dtype=np.float32)
        view_pos = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        color = np.array([0.0, 0.682, 0.937], dtype=np.float32)

        prog.set_vec3("uLightPos", light_pos)
        prog.set_vec3("uViewPos", view_pos)
        prog.set_vec3("uColor", color)
        prog.set_float("uGlowIntensity", config.render.glow_intensity)
        prog.set_float("uTime", self._time)

        if shader_key == "wireframe":
            GL.glLineWidth(1.5)

        GL.glBindVertexArray(mesh.vao)
        GL.glDrawElements(mesh.primitive_type, mesh.index_count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        if shader_key == "wireframe":
            GL.glLineWidth(1.0)

    def cleanup(self):
        for mesh in self._meshes.values():
            mesh.delete()
        self._meshes.clear()
        for prog in self._shaders.values():
            prog.delete()
        self._shaders.clear()
        self._initialized = False
        logger.info("Renderer cleaned up")
