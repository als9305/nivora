"""Compatibility package for the documented ``backend.app.main:app`` entry point.

The implementation historically imports application modules as ``app.*``.
Putting this directory on the import path before Uvicorn imports ``backend.app``
keeps that namespace stable for both the documented command and test runner.
"""

import os
import sys

_backend_dir = os.path.dirname(__file__)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
