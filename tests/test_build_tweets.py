import json
from pathlib import Path

from foundry.datasets.build_tweets import (
    BuildReport,
    build_labeled_tweets,
    dedupe,
    split_for_id,
    strip_pii,
    write_splits,
)


def _lines(fixtures_dir: Path) -> list[str]:
    return (fixtures_dir / 'traces_fixture.jsonl').read_text(encoding='utf-8').splitlines()


def test_near_duplicates_collapse_keeping_first() -> None:
    items = [
        ('a', 'Dario Ketteridge ruled out for Harbor City with a hamstring injury.'),
        ('b', 'Dario Ketteridge ruled out for Harbor City with a hamstring injury. #news'),
        ('c', 'Pre match show starts at seven tonight on the club channel.'),
    ]
    assert [item_id for item_id, _ in dedupe(items)] == ['a', 'c']


def test_split_is_stable_and_roughly_proportioned() -> None:
    assert split_for_id('fx-001') == split_for_id('fx-001')
    splits = [split_for_id(f'id-{i}') for i in range(2000)]
    assert 0.74 < splits.count('train') / 2000 < 0.86
    assert splits.count('dev') > 0 and splits.count('test') > 0


def test_pii_is_stripped() -> None:
    text = 'mail desk@oddsmith.dev or +44 20 7946 0958 via @sidelinescout https://news.oddsmith.dev/a1'
    cleaned = strip_pii(text)
    for leaked in ('desk@', '7946', '@sidelinescout', 'https://'):
        assert leaked not in cleaned
    assert '[email]' in cleaned and '[phone]' in cleaned and '[handle]' in cleaned


def test_build_from_fixture_traces(fixtures_dir: Path) -> None:
    report = BuildReport()
    rows = build_labeled_tweets(_lines(fixtures_dir), report)
    ids = {tweet.id for _, tweet in rows}
    assert 't-dup' not in ids
    assert 't-nolabel' not in ids
    assert report.invalid == 1
    assert report.unlabeled == 1
    assert report.duplicates == 1
    assert report.kept == len(rows) == sum(report.splits.values())
    pii_row = next(t for _, t in rows if t.id == 't-pii')
    assert 'oddsmith.dev' not in pii_row.text
    assert pii_row.label_source in {'teacher', 'human'}


def test_write_splits_is_deterministic(fixtures_dir: Path, tmp_path: Path) -> None:
    rows = build_labeled_tweets(_lines(fixtures_dir))
    first = write_splits(rows, tmp_path / 'one')
    second = write_splits(rows, tmp_path / 'two')
    for split in first:
        assert first[split].read_text() == second[split].read_text()
    total = sum(len(p.read_text().splitlines()) for p in first.values())
    assert total == len(rows)
    assert json.loads(next(iter(first.values())).read_text().splitlines()[0])['id']
