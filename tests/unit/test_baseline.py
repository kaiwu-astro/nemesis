"""Baseline unit tests for project scaffolding."""

from pathlib import Path


def test_pyproject_contains_baseline_tooling_configuration():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in pyproject_text
    assert '"tests/unit"' in pyproject_text
    assert 'target-version = "py310"' in pyproject_text
