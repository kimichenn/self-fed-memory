"""
Shared pytest fixtures for all tests.
"""

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

# ---------------------------------------------------------------------------
# Automatic test markers based on path
# ---------------------------------------------------------------------------
# We want to avoid sprinkling `@pytest.mark.unit` / `integration` / `manual`
# decorators throughout the codebase.  Instead, assign the marker implicitly
# from the directory the test file lives in.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Dynamically add pytest markers depending on filepath.

    Any test located in ``tests/unit`` gets the ``unit`` marker,
    tests in ``tests/integration`` get ``integration``, and tests in
    ``tests/manual`` get ``manual``.

    This allows developers to drop explicit decorators in test source files
    while retaining the same marker-based selection semantics (e.g.
    ``pytest -m unit``).
    """

    root_path = Path(config.rootdir)

    for item in items:
        # Convert the file path to a string relative to the project root
        rel_path = Path(item.fspath).resolve().relative_to(root_path).as_posix()

        if rel_path.startswith("tests/unit/"):
            item.add_marker("unit")
        elif rel_path.startswith("tests/integration/"):
            item.add_marker("integration")
        elif rel_path.startswith("tests/manual/"):
            item.add_marker("manual")


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped monkeypatch fixture.

    This behaves like pytest's built-in ``monkeypatch`` fixture but lives for the
    entire module so state set up in one test can be reused by others (helpful
    for expensive environment variable or network patching common in manual
    verification tests).
    """
    mpatch = MonkeyPatch()
    try:
        yield mpatch
    finally:
        mpatch.undo()
