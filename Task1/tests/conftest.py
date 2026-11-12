"""pytest 公共 fixture —— 把 Task1/ 根目录加入 sys.path，使 tests/ 下能直接 import tin_coating 顶层包。"""
import sys
from pathlib import Path
import pytest

TASK1_ROOT = Path(__file__).resolve().parent.parent
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))


@pytest.fixture(scope="session")
def task1_root() -> Path:
    return TASK1_ROOT


@pytest.fixture(scope="session")
def featured_data_path(task1_root) -> Path:
    return task1_root / "result" / "data" / "feature_engineered_data" / "featured_data.xlsx"


@pytest.fixture(scope="session")
def prod_config_path(task1_root) -> Path:
    return task1_root / "group_params_2_799_only.json"
