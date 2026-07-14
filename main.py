import sys
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hologesture.log", mode="w"),
    ]
)

logger = logging.getLogger(__name__)

os.environ["QT_OPENGL"] = "desktop"
os.environ["QT_OPENGL_NO_MIPMAP"] = "1"


def main():
    from PyQt5.QtWidgets import QApplication
    from app.main_window import HoloGestureWindow
    from app.errors import install_global_exception_hook

    install_global_exception_hook()

    app = QApplication(sys.argv)
    app.setApplicationName("HoloGesture")
    app.setApplicationVersion("1.0.0")

    window = HoloGestureWindow()
    window.show()

    logger.info("HoloGesture started")
    exit_code = app.exec_()
    logger.info("HoloGesture exited")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
