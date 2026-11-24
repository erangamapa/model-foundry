"""Turn a trace export into labeled tweet jsonl.

Each trace line carries the news item exactly as the ingest service saw it, plus the
label that came back from a teacher model or a human reviewer.
"""

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sportsbook_contracts import NewsItem

from foundry.datasets.schemas import (
    Certainty,
    EventType,
    LabeledTweet,
    LabelSource,
    MarketDirection,
)

Split = Literal['train', 'dev', 'test']

EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
PHONE = re.compile(r'(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)')
HANDLE = re.compile(r'(?<![\w@])@\w{2,}')
URL = re.compile(r'https?://\S+')

SHINGLE_SIZE = 3
MINHASH_ROWS = 24
BAND_SIZE = 3
TRAIN_CUTOFF = 80
DEV_CUTOFF = 90


class TraceLabel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    event_type: EventType
    entity: str = Field(min_length=1)
    certainty: Certainty
    market_direction: MarketDirection


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    trace_id: str = Field(alias='traceId', min_length=1)
    news_item: NewsItem = Field(alias='newsItem')
    label: TraceLabel | None = None
    label_source: LabelSource | None = Field(alias='labelSource', default=None)


@dataclass
class BuildReport:
    read: int = 0
    unlabeled: int = 0
    invalid: int = 0
    duplicates: int = 0
    kept: int = 0
    splits: dict[str, int] = field(default_factory=lambda: {'train': 0, 'dev': 0, 'test': 0})


def strip_pii(text: str) -> str:
    redacted = EMAIL.sub('[email]', text)
    redacted = URL.sub('[link]', redacted)
    redacted = PHONE.sub('[phone]', redacted)
    return HANDLE.sub('[handle]', redacted)


def _normalize(text: str) -> list[str]:
    cleaned = URL.sub(' ', text.lower())
    cleaned = HANDLE.sub(' ', cleaned)
    return re.sub(r'[^a-z0-9\s]', ' ', cleaned).split()


def _shingles(tokens: list[str]) -> set[str]:
    if len(tokens) < SHINGLE_SIZE:
        return {' '.join(tokens)}
    return {' '.join(tokens[i : i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1)}


def _hash64(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f'{seed}:{value}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big')


def minhash_buckets(text: str) -> list[str]:
    """Banded minhash keys. Two texts sharing any key count as near duplicates."""
    shingles = _shingles(_normalize(text))
    signature = [min(_hash64(seed, s) for s in shingles) for seed in range(MINHASH_ROWS)]
    return [
        f'{band}:{signature[band * BAND_SIZE : (band + 1) * BAND_SIZE]}'
        for band in range(MINHASH_ROWS // BAND_SIZE)
    ]


def dedupe(texts: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Keep the first item seen for every near duplicate group. Items are (id, text)."""
    seen: set[str] = set()
    for item_id, text in texts:
        buckets = minhash_buckets(text)
        if any(bucket in seen for bucket in buckets):
            continue
        seen.update(buckets)
        yield item_id, text


def split_for_id(item_id: str) -> Split:
    slot = int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], 'big') % 100
    if slot < TRAIN_CUTOFF:
        return 'train'
    return 'dev' if slot < DEV_CUTOFF else 'test'


def parse_traces(lines: Iterable[str], report: BuildReport) -> Iterator[TraceRecord]:
    for line in lines:
        if not line.strip():
            continue
        report.read += 1
        try:
            yield TraceRecord.model_validate_json(line)
        except ValidationError:
            report.invalid += 1


def build_labeled_tweets(
    lines: Iterable[str], report: BuildReport | None = None
) -> list[tuple[Split, LabeledTweet]]:
    report = report or BuildReport()
    labeled: list[TraceRecord] = []
    for record in parse_traces(lines, report):
        if record.label is None or record.label_source is None:
            report.unlabeled += 1
            continue
        labeled.append(record)

    unique_ids = {
        item_id
        for item_id, _ in dedupe((r.trace_id, r.news_item.text) for r in labeled)
    }
    report.duplicates = len(labeled) - len(unique_ids)

    rows: list[tuple[Split, LabeledTweet]] = []
    for record in labeled:
        if record.trace_id not in unique_ids or record.label is None or record.label_source is None:
            continue
        tweet = LabeledTweet(
            id=record.trace_id,
            text=strip_pii(record.news_item.text),
            source=record.news_item.source,
            event_type=record.label.event_type,
            entity=record.label.entity,
            certainty=record.label.certainty,
            market_direction=record.label.market_direction,
            label_source=record.label_source,
        )
        split = split_for_id(tweet.id)
        report.splits[split] += 1
        rows.append((split, tweet))
    report.kept = len(rows)
    return rows


def write_splits(rows: list[tuple[Split, LabeledTweet]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: out_dir / f'tweets_{split}.jsonl' for split in ('train', 'dev', 'test')}
    for split, path in paths.items():
        with path.open('w', encoding='utf-8') as handle:
            for row_split, tweet in rows:
                if row_split == split:
                    handle.write(json.dumps(tweet.model_dump(), ensure_ascii=False) + '\n')
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Build labeled tweet splits from a trace export')
    parser.add_argument('traces', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    report = BuildReport()
    with args.traces.open(encoding='utf-8') as handle:
        rows = build_labeled_tweets(handle, report)
    write_splits(rows, args.out_dir)
    print(json.dumps(report.__dict__))


if __name__ == '__main__':
    main()
