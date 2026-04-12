from __future__ import annotations

import importlib
import sys
from pathlib import Path

from core.config import BASE_DIR

VENDOR_DIR = BASE_DIR / ".vendor"


def ensure_vendor_path() -> Path:
    vendor_path = str(VENDOR_DIR)
    if VENDOR_DIR.exists() and vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    return VENDOR_DIR


def import_vendor_module(module_name: str):
    ensure_vendor_path()
    return importlib.import_module(module_name)
