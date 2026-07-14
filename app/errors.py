import logging
import traceback
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    CAMERA_UNAVAILABLE = auto()
    CAMERA_DISCONNECTED = auto()
    TRACKING_FAILURE = auto()
    RENDERER_FAILURE = auto()
    SHADER_COMPILE_ERROR = auto()
    LOW_FPS = auto()
    NO_HAND_DETECTED = auto()
    GESTURE_AMBIGUITY = auto()
    APP_CRASH = auto()
    CONFIG_ERROR = auto()


class HoloError(Exception):
    def __init__(self, code: ErrorCode, message: str, recoverable: bool = True):
        self.code = code
        self.message = message
        self.recoverable = recoverable
        super().__init__(self.message)

    def log(self):
        logger.error(f"[{self.code.name}] {self.message}")
        logger.debug(traceback.format_exc())


class CameraError(HoloError):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(ErrorCode.CAMERA_UNAVAILABLE, message, recoverable)


class TrackingError(HoloError):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(ErrorCode.TRACKING_FAILURE, message, recoverable)


class RendererError(HoloError):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(ErrorCode.RENDERER_FAILURE, message, recoverable)


class ShaderError(HoloError):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(ErrorCode.SHADER_COMPILE_ERROR, message, recoverable)


def install_global_exception_hook():
    def exception_hook(exc_type, exc_value, exc_tb):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        error = HoloError(ErrorCode.APP_CRASH, f"Unhandled exception: {exc_value}", recoverable=False)
        error.log()

    import sys
    sys.excepthook = exception_hook
