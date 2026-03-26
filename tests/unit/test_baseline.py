"""Baseline unit tests for project scaffolding."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_pyproject_contains_baseline_tooling_configuration():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.10"
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/unit"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py310"
