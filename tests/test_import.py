from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import epanet_rl  # noqa: E402


def test_package_importable() -> None:
    assert epanet_rl.__version__ == "0.1.0"


def test_networks_dir_exists() -> None:
    assert epanet_rl.NETWORKS_DIR.exists()
