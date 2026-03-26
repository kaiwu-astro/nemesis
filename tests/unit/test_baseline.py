"""Baseline unit tests for project scaffolding."""


def test_unit_test_scaffold_is_active():
    assert True


def test_runtime_requirements_list_exists_and_has_expected_basics():
    with open("requirements.txt", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read()

    assert "numpy" in requirements
    assert "AMUSE community packages are installed separately" in requirements
