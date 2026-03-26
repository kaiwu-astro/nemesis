"""Baseline unit tests for project scaffolding."""

from pathlib import Path

import tomllib


def test_unit_test_scaffold_is_active():
    assert True


def test_pyproject_contains_baseline_tooling_configuration():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.10"
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/unit"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py310"
