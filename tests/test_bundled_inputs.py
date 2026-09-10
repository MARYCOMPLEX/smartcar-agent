"""A fresh checkout must contain real, unmodified assets, including proxy-only wheels."""
import importlib.util
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_install", ROOT / "tools/check_install.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_checkout_includes_all_eight_verified_geometry_assets():
    records = CHECK.check_inputs(ROOT)
    assert len(records) == 8
    assert sum('/exact/' in record['path'] for record in records) == 5
    assert sum('/proxy/' in record['path'] for record in records) == 2


def test_corrupt_bundled_asset_is_rejected_before_native_import(tmp_path):
    shutil.copytree(ROOT / CHECK.DATA, tmp_path / CHECK.DATA)
    sample = tmp_path / CHECK.DATA / "appearance/vehicle_appearance.stl"
    content = bytearray(sample.read_bytes())
    content[-1] ^= 1
    sample.write_bytes(content)
    with pytest.raises(ValueError, match='checksum mismatch'):
        CHECK.check_inputs(tmp_path)
