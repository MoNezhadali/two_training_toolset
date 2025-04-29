"""Configuration for unit tests."""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_getenv():
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: (
            "src/serve/config.yaml" if key == "APP_CONFIG_PATH" else default
        )
        yield
