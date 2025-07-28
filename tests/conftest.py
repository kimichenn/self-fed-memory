"""
Shared pytest fixtures for all tests.
"""

import os
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

from app.core.config import Settings

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


# ---------------------------------------------------------------------------
# Global environment setup for Pinecone tests
# ---------------------------------------------------------------------------
# Some tests instantiate the production `PineconeVectorStore` (either directly
# or indirectly via `MemoryManager`).  To guarantee these tests **always** use
# a dedicated *test* index – and never accidentally read from / write to a
# production collection – we force the `PINECONE_INDEX` environment variable to
# the value provided by our testing settings.  Because we register the fixture
# with ``autouse=True`` it applies automatically to *every* test module.


@pytest.fixture(autouse=True, scope="session")
def _ensure_test_pinecone_index():
    """Ensure all tests use the *test* Pinecone index.

    We first look for an explicit ``TEST_PINECONE_INDEX`` environment variable
    (which individual tests *may* set for customisation).  If absent, we fall
    back to the default provided by ``Settings.for_testing()``.  The chosen
    value is then injected into ``PINECONE_INDEX`` so that any subsequent
    instantiation of ``Settings()`` - for example in the production
    ``PineconeVectorStore`` - resolves to the correct index.
    """

    test_index = os.getenv("TEST_PINECONE_INDEX", Settings.for_testing().pinecone_index)

    # Apply environment patch at session scope using our own MonkeyPatch instance
    mpatch = MonkeyPatch()
    mpatch.setenv("PINECONE_INDEX", test_index)

    # Yield control to the test session
    try:
        yield
    finally:
        # Roll back after all tests complete
        mpatch.undo()
