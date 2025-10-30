
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from tests.helpers import DummyConfig, DummyModel


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests to use."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def mock_wallet():
    """Provides a mock bittensor wallet."""
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "fake_hotkey_address"
    return wallet


@pytest.fixture
def dummy_model():
    """Provides a dummy model instance."""
    config = DummyConfig()
    return DummyModel(config)
