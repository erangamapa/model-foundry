"""Split deepset/prompt-injections into train, dev and test jsonl under data/splits.

The dataset ships its own train and test split. The test split is kept untouched and a
stratified 12 percent of train becomes dev. Needs the train extra and network access.
"""
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT = Path('data/splits')


def to_rows(split) -> list[dict]:
    return [{'text': row['text'], 'label': int(row['label'])} for row in split]


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / f'{name}.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(name, len(rows), sum(row['label'] for row in rows))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset('deepset/prompt-injections')
    train = to_rows(dataset['train'])
    train_part, dev = train_test_split(
        train, test_size=0.12, random_state=13, stratify=[row['label'] for row in train]
    )
    write('injection_train', train_part)
    write('injection_dev', dev)
    write('injection_test', to_rows(dataset['test']))


if __name__ == '__main__':
    main()
