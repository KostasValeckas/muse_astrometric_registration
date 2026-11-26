"""Smoke tests: the package modules import and expose their main classes and the
entry points of the commands."""

import importlib

import pytest


PACKAGE = "muse_astrometric_registration"


def test_methods_exports_no_alignment():
    """methods exports a callable NoAlignment."""
    methods = importlib.import_module(f"{PACKAGE}.methods")
    assert hasattr(methods, "NoAlignment")
    assert callable(methods.NoAlignment)


def test_pipeline_can_import():
    """pipeline imports and defines AlignmentPipeline."""
    pipeline = importlib.import_module(f"{PACKAGE}.pipeline")
    assert hasattr(pipeline, "AlignmentPipeline")


@pytest.mark.parametrize("module", ["hybrid_pipeline", "run", "manual_alignment"])
def test_command_entry_points(module):
    """The modules of the commands (setup.cfg, console_scripts) define main()."""
    imported = importlib.import_module(f"{PACKAGE}.{module}")
    assert callable(imported.main)
