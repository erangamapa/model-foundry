import shutil
from pathlib import Path

import pytest

from foundry.datasets.golden import GoldenDriftError, freeze, load, manifest_path


@pytest.fixture
def golden_copy(root: Path, tmp_path: Path) -> Path:
    target = tmp_path / 'golden.jsonl'
    shutil.copy(root / 'data' / 'golden' / 'tweets_golden_sample.jsonl', target)
    return target


def test_committed_sample_verifies(root: Path) -> None:
    rows = load(root / 'data' / 'golden' / 'tweets_golden_sample.jsonl')
    assert len(rows) == 30


def test_freeze_then_load(golden_copy: Path) -> None:
    manifest = freeze(golden_copy)
    assert manifest.rows == 30
    assert len(load(golden_copy)) == 30


def test_drifted_file_is_refused(golden_copy: Path) -> None:
    freeze(golden_copy)
    golden_copy.write_text(golden_copy.read_text().replace('injury', 'result', 1))
    with pytest.raises(GoldenDriftError):
        load(golden_copy)


def test_refreeze_of_changed_file_is_refused(golden_copy: Path) -> None:
    freeze(golden_copy)
    golden_copy.write_text(golden_copy.read_text().replace('injury', 'result', 1))
    with pytest.raises(GoldenDriftError):
        freeze(golden_copy)


def test_missing_manifest_is_refused(golden_copy: Path) -> None:
    assert not manifest_path(golden_copy).exists()
    with pytest.raises(GoldenDriftError):
        load(golden_copy)
