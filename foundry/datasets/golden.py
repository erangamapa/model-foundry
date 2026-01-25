import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from foundry.datasets.schemas import LabeledTweet


class GoldenDriftError(Exception):
    """The golden file no longer matches its manifest."""


class GoldenManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sha256: str
    rows: int


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + '.manifest.json')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(path: Path) -> list[LabeledTweet]:
    with path.open(encoding='utf-8') as handle:
        return [LabeledTweet.model_validate_json(line) for line in handle if line.strip()]


def freeze(path: Path) -> GoldenManifest:
    """Write the manifest for a golden file. A frozen set can never be silently replaced."""
    rows = _parse(path)
    manifest = GoldenManifest(sha256=_digest(path), rows=len(rows))
    target = manifest_path(path)
    if target.exists():
        existing = GoldenManifest.model_validate_json(target.read_text(encoding='utf-8'))
        if existing != manifest:
            raise GoldenDriftError(f'{path.name} is already frozen with a different hash')
        return existing
    target.write_text(json.dumps(manifest.model_dump(), indent=2) + '\n', encoding='utf-8')
    return manifest


def load(path: Path) -> list[LabeledTweet]:
    target = manifest_path(path)
    if not target.exists():
        raise GoldenDriftError(f'{path.name} has no manifest, freeze it first')
    manifest = GoldenManifest.model_validate_json(target.read_text(encoding='utf-8'))
    actual = _digest(path)
    if actual != manifest.sha256:
        raise GoldenDriftError(
            f'{path.name} drifted, expected {manifest.sha256[:12]} got {actual[:12]}'
        )
    rows = _parse(path)
    if len(rows) != manifest.rows:
        raise GoldenDriftError(f'{path.name} row count changed')
    return rows
