from pathlib import Path

import pytest

from foundry.hub.push import PushRefusedError, assert_safetensors_only, collect_upload_files, push


def _adapter(tmp_path: Path, *names: str) -> Path:
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in names:
        (adapter / name).write_bytes(b'x')
    return adapter


@pytest.fixture
def card_and_config(tmp_path: Path) -> tuple[Path, Path]:
    card, config = tmp_path / 'card.md', tmp_path / 'cfg.yaml'
    card.write_text('card')
    config.write_text('a: 1')
    return card, config


def test_safetensors_adapter_is_allowed(tmp_path: Path, card_and_config) -> None:
    adapter = _adapter(tmp_path, 'adapter_model.safetensors', 'adapter_config.json')
    uploads = collect_upload_files(adapter, *card_and_config)
    assert set(uploads) == {
        'adapter_model.safetensors', 'adapter_config.json', 'README.md', 'train_config.yaml',
    }


@pytest.mark.parametrize('bad', ['pytorch_model.bin', 'weights.pt', 'x.pkl', 'm.joblib', 'a.CKPT'])
def test_pickle_formats_are_rejected(tmp_path: Path, card_and_config, bad: str) -> None:
    adapter = _adapter(tmp_path, 'adapter_model.safetensors', bad)
    with pytest.raises(PushRefusedError):
        collect_upload_files(adapter, *card_and_config)


def test_adapter_without_weights_is_rejected(tmp_path: Path, card_and_config) -> None:
    adapter = _adapter(tmp_path, 'adapter_config.json')
    with pytest.raises(PushRefusedError):
        collect_upload_files(adapter, *card_and_config)


def test_guard_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / 'folder.bin').mkdir()
    assert_safetensors_only([tmp_path / 'folder.bin'])


def test_push_needs_token_before_any_network(
    tmp_path: Path, card_and_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('HF_TOKEN', raising=False)
    adapter = _adapter(tmp_path, 'adapter_model.safetensors')
    with pytest.raises(PushRefusedError):
        push('oddsmith/x', adapter, *card_and_config)
