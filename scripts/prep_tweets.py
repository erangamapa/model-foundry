"""Build tweet train, dev and test jsonl under data/splits from the committed samples.

Train and dev come from the synthetic fixture. The frozen golden sample is the test set, and
any fixture row whose text also appears in the golden sample is dropped first.
"""
import csv
import json
import random
from pathlib import Path

OUT = Path('data/splits')
KEYS = [
    'id', 'text', 'source', 'event_type', 'entity', 'certainty', 'market_direction', 'label_source',
]


def normalize(text: str) -> str:
    return ' '.join(text.lower().split())


def pick(row: dict) -> dict:
    return {key: row[key] for key in KEYS}


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / f'{name}.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(name, len(rows))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open('data/fixtures/tweets_fixture.csv', encoding='utf-8') as handle:
        fixture = list(csv.DictReader(handle))
    with open('data/golden/tweets_golden_sample.jsonl', encoding='utf-8') as handle:
        golden = [json.loads(line) for line in handle]
    golden_texts = {normalize(row['text']) for row in golden}
    fixture = [pick(row) for row in fixture if normalize(row['text']) not in golden_texts]
    random.Random(13).shuffle(fixture)
    write('tweets_train', fixture[10:])
    write('tweets_dev', fixture[:10])
    write('tweets_test', [pick(row) for row in golden])


if __name__ == '__main__':
    main()
