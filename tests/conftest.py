from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_psd_path() -> Path:
    return FIXTURES / "mini_model.psd"
