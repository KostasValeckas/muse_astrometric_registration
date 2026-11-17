"""Smoke tests: the modules in src/ import and expose their main classes."""

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_methods_exports_no_alignment():
    """methods exports a callable NoAlignment."""
    methods = importlib.import_module("methods")
    assert hasattr(methods, "NoAlignment")
    assert callable(methods.NoAlignment)


def test_pipeline_can_import():
    """pipeline imports and defines AlignmentPipeline."""
    pipeline = importlib.import_module("pipeline")
    assert hasattr(pipeline, "AlignmentPipeline")
