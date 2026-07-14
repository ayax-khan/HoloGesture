import logging
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
from OpenGL import GL

from config import config
from app.object_model import ObjectModel
from app.math_utils import perspective_matrix, look_at_matrix
from app.mesh import Mesh, ObjectType, MODEL_FILES
from app.shaders import SHADER_SOURCES, VERTEX_SHADER_SRC
from app.obj_loader import load_obj, ObjModel
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


class LoadedModelManager:
    def __init__(self, models_dir: str):
        self._models_dir = models_dir
        self._obj_data: Dict[str, Optional[ObjModel]] = {}
        self._model_names: List[str] = []
        self._current_index: int = 0

    def scan(self):
        if not os.path.isdir(self._models_dir):
            logger.warning("Models directory not found: %s", self._models_dir)
            return
        self._model_names = []
        for fname in sorted(os.listdir(self._models_dir)):
            if fname.lower().endswith(".obj"):
                self._model_names.append(fname)
                path = os.path.join(self._models_dir, fname)
                self._obj_data[fname] = load_obj(path)
        logger.info("Found %d OBJ models in %s", len(self._model_names), self._models_dir)

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    def get_current(self) -> Optional[ObjModel]:
        if not self._model_names:
            return None
        name = self._model_names[self._current_index]
        return self._obj_data.get(name)

    def get_current_name(self) -> str:
        if not self._model_names:
            return "No Model"
        return self._model_names[self._current_index].replace(".obj", "").capitalize()

    def next(self):
        if self._model_names:
            self._current_index = (self._current_index + 1) % len(self._model_names)

    def prev(self):
        if self._model_names:
            self._current_index = (self._current_index - 1) % len(self._model_names)


class Renderer:
    SHADER_KEYS = {"solid": "solid", "wireframe": "wireframe", "holographic": "holographic"}

    def __init__(self):
        self._shaders: Dict[str, ShaderProgram] = {}
        self._meshes: Dict[ObjectType, Mesh] = {}
        self._loaded_mesh: Optional[Mesh] = None
        self._initialized = False
        self._time = 0.0
        self._view_matrix: Optional[np.ndarray] = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._current_obj_type = ObjectType.SOLID_CUBE
        self._load_next = False
        self._camera_orbit_angle = 0.0
        self._camera_height = 1.0
        self._camera_distance = 3.5
        self._model_manager = LoadedModelManager(
            os.path.join(config.app.assets_dir, "models")
        )

    def initialize(self):
        logger.info("Initializing renderer")
        try:
            self._build_shaders()
            self._build_meshes()
            self._model_manager.scan()
            self._rebuild_loaded_mesh()
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
            if obj_type == ObjectType.LOADED_MODEL:
                continue
            mesh = Mesh()
            mesh.build(obj_type, config.render.mesh_detail)
            self._meshes[obj_type] = mesh

    def _rebuild_loaded_mesh(self):
        if self._loaded_mesh:
            self._loaded_mesh.delete()
            self._loaded_mesh = None
        obj_model = self._model_manager.get_current()
        if obj_model is not None:
            self._loaded_mesh = Mesh()
            self._loaded_mesh.build_from_data(
                obj_model.vertices, obj_model.indices
            )



    @property
    def current_object_type(self) -> ObjectType:
        return self._current_obj_type

    @current_object_type.setter
    def current_object_type(self, obj_type: ObjectType):
        if obj_type == ObjectType.LOADED_MODEL:
            self._model_manager.next()
            self._rebuild_loaded_mesh()
        elif obj_type in self._meshes:
            self._current_obj_type = obj_type
            logger.info(f"Switched to {obj_type.name}")

    @property
    def current_model_name(self) -> str:
        return self._model_manager.get_current_name()

    @property
    def has_loaded_model(self) -> bool:
        return self._loaded_mesh is not None

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

        mesh: Optional[Mesh] = None
        shader_key = "solid"

        if self._current_obj_type == ObjectType.LOADED_MODEL and self._loaded_mesh:
            mesh = self._loaded_mesh
            shader_key = "solid"
        elif self._current_obj_type in self._meshes:
            mesh = self._meshes[self._current_obj_type]
            if self._current_obj_type == ObjectType.WIREFRAME_CUBE:
                shader_key = "wireframe"
            elif self._current_obj_type == ObjectType.HOLOGRAPHIC_SPHERE:
                shader_key = "holographic"

        if mesh is not None:
            prog = self._shaders[shader_key]

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
        if self._loaded_mesh:
            self._loaded_mesh.delete()
            self._loaded_mesh = None
        for mesh in self._meshes.values():
            mesh.delete()
        self._meshes.clear()
        for prog in self._shaders.values():
            prog.delete()
        self._shaders.clear()
        self._initialized = False
        logger.info("Renderer cleaned up")
