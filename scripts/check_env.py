#!/usr/bin/env python3
"""Environment diagnostics for NEMESIS.

This script validates whether the current Python environment can run different
levels of repository workflows:
- base-dev: linting and unit tests only
- amuse-dev: AMUSE import + core community workers
- full-regression: amuse-dev + compiled C++ extension availability
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
WORKER_SO = ROOT / "src" / "build" / "kick_particles_worker.so"
MIN_PYTHON = (3, 10)


@dataclass
class CheckResult:
    """Single check result for display/reporting."""

    name: str
    passed: bool
    details: str
    remediation: str | None = None


def check_python_version() -> CheckResult:
    """Check minimum supported Python version."""
    current = sys.version_info[:3]
    passed = current >= MIN_PYTHON
    details = f"Detected Python {platform.python_version()}"
    remediation = None
    if not passed:
        remediation = (
            f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and recreate the environment."
        )
    return CheckResult("Python version", passed, details, remediation)


def check_module(module_name: str) -> CheckResult:
    """Check whether a Python module can be imported."""
    try:
        importlib.import_module(module_name)
        return CheckResult(
            f"Module import: {module_name}",
            True,
            f"{module_name} import succeeded",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic script should capture all import failures
        remediation = (
            "Install missing dependencies via `conda env update -f environment.yml` "
            "or ensure this package exists in your active environment."
        )
        return CheckResult(
            f"Module import: {module_name}",
            False,
            f"{module_name} import failed: {exc}",
            remediation,
        )


def check_amuse_code(code_name: str) -> CheckResult:
    """Check if an AMUSE community code class can be imported."""
    module_path = f"amuse.community.{code_name.lower()}.interface"
    try:
        module = importlib.import_module(module_path)
        getattr(module, code_name)
        return CheckResult(
            f"AMUSE code available: {code_name}",
            True,
            f"Found {code_name} in {module_path}",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic script should capture all import failures
        remediation = (
            "Install AMUSE community packages (e.g. amuse-huayno, amuse-ph4, "
            "amuse-seba, amuse-symple, amuse-kepler) in this environment."
        )
        return CheckResult(
            f"AMUSE code available: {code_name}",
            False,
            f"Could not load {code_name} from {module_path}: {exc}",
            remediation,
        )


def check_worker_library() -> CheckResult:
    """Check presence of compiled C++ worker shared object."""
    passed = WORKER_SO.exists()
    details = f"Checked path: {WORKER_SO}"
    remediation = None
    if not passed:
        remediation = "Compile the C++ worker with `cd src/cpp && make`."
    return CheckResult("C++ worker library", passed, details, remediation)


def run_checks() -> list[CheckResult]:
    """Execute all checks and return ordered results."""
    checks: list[Callable[[], CheckResult]] = [
        check_python_version,
        lambda: check_module("numpy"),
        lambda: check_module("natsort"),
        lambda: check_module("h5py"),
        lambda: check_module("tables"),
        lambda: check_module("amuse"),
        lambda: check_amuse_code("Huayno"),
        lambda: check_amuse_code("Ph4"),
        lambda: check_amuse_code("SeBa"),
        lambda: check_amuse_code("Symple"),
        lambda: check_amuse_code("Kepler"),
        check_worker_library,
    ]
    return [fn() for fn in checks]


def print_report(results: list[CheckResult]) -> int:
    """Render a readable report and return process exit code."""
    print("NEMESIS environment diagnostics")
    print("=" * 32)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"       {result.details}")
        if result.remediation and not result.passed:
            print(f"       Remediation: {result.remediation}")

    base_dev_targets = {
        "Python version",
        "Module import: numpy",
        "Module import: natsort",
        "Module import: h5py",
        "Module import: tables",
    }
    base_dev_ok = all(r.passed for r in results if r.name in base_dev_targets)

    amuse_dev_ok = all(
        r.passed
        for r in results
        if r.name == "Module import: amuse" or r.name.startswith("AMUSE code available:")
    )

    full_regression_ok = amuse_dev_ok and next(
        r for r in results if r.name == "C++ worker library"
    ).passed

    print("\nEnvironment tiers")
    print("-" * 18)
    print(f"base-dev        : {'READY' if base_dev_ok else 'NOT READY'}")
    print(f"amuse-dev       : {'READY' if amuse_dev_ok else 'NOT READY'}")
    print(f"full-regression : {'READY' if full_regression_ok else 'NOT READY'}")

    if full_regression_ok:
        print("\nOverall status: PASS (full scientific workflow available)")
        return 0
    if base_dev_ok:
        print("\nOverall status: PARTIAL (base development workflow available)")
        return 1

    print("\nOverall status: FAIL (core dependencies missing)")
    return 2


if __name__ == "__main__":
    raise SystemExit(print_report(run_checks()))
