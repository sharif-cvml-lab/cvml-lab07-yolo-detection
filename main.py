from __future__ import annotations

import sys

# onnxruntime must be imported before PyQt6: on some Windows systems loading Qt first makes
# the onnxruntime DLL fail to initialise.
import onnxruntime  # noqa: F401
from gui import AppWindow, create_application


def main() -> int:
    app = create_application()
    window = AppWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
