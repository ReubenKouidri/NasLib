from pathlib import Path

import pytest

from dnasty.data.splits import resolve_data_dir
from dnasty.utils import Config, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture(autouse=True)
def _seed():
    seed_everything(0)


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    return Config.from_file(CONFIGS / "tiny.yaml").nas


@pytest.fixture(scope="session")
def cpsc_paths(tiny_config):
    """(data_dir, reference_path) for the local CPSC subset, or skip."""
    root = resolve_data_dir(tiny_config)
    if not root.is_absolute():
        root = REPO_ROOT / root
    data_dir = root / tiny_config.data.subdir
    reference = root / tiny_config.data.reference
    if not data_dir.is_dir() or not reference.is_file():
        pytest.skip(f"CPSC data not found under {root} (set DNASTY_DATA_DIR)")
    return data_dir, reference
