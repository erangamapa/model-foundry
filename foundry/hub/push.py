"""Publish an adapter to the Hub. Safetensors only, pickle formats are refused."""

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_SUFFIXES = frozenset(
    {'.bin', '.pt', '.pth', '.pkl', '.pickle', '.ckpt', '.joblib', '.npy', '.npz'}
)
HUB_URL = 'https://huggingface.co'


class PushRefusedError(Exception):
    pass


def assert_safetensors_only(paths: Iterable[Path]) -> None:
    bad = sorted(p.name for p in paths if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES)
    if bad:
        raise PushRefusedError(f'refusing non safetensors files, {", ".join(bad)}')


def collect_upload_files(adapter_dir: Path, card: Path, config: Path) -> dict[str, Path]:
    """Map of repo path to local file. Raises before anything is uploaded."""
    adapter_files = sorted(p for p in adapter_dir.rglob('*') if p.is_file())
    assert_safetensors_only(adapter_files)
    if not any(p.suffix == '.safetensors' for p in adapter_files):
        raise PushRefusedError(f'no .safetensors weights in {adapter_dir}')
    uploads = {str(p.relative_to(adapter_dir)): p for p in adapter_files}
    for doc in (card, config):
        if doc.suffix.lower() not in {'.md', '.yaml', '.yml'}:
            raise PushRefusedError(f'refusing unexpected doc file type, {doc.name}')
    uploads['README.md'] = card
    uploads['train_config.yaml'] = config
    return uploads


def push(repo_id: str, adapter_dir: Path, card: Path, config: Path, private: bool = True) -> str:
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise PushRefusedError('HF_TOKEN is not set')
    uploads = collect_upload_files(adapter_dir, card, config)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='model', private=private, exist_ok=True)
    for repo_path, local_path in uploads.items():
        api.upload_file(
            path_or_fileobj=str(local_path), path_in_repo=repo_path, repo_id=repo_id
        )
    url = f'{HUB_URL}/{repo_id}'
    print(url)
    for repo_path in uploads:
        print(f'{url}/blob/main/{repo_path}')
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description='Push an adapter, card and config to the Hub')
    parser.add_argument('repo_id')
    parser.add_argument('adapter_dir', type=Path)
    parser.add_argument('card', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('--public', action='store_true')
    args = parser.parse_args()
    push(args.repo_id, args.adapter_dir, args.card, args.config, private=not args.public)


if __name__ == '__main__':
    main()
