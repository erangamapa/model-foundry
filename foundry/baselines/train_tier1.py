"""Tier one baselines. Logistic regression always, xgboost when it is installed."""

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from foundry.baselines.features import FeatureRecord, TweetFeaturizer
from foundry.datasets.build_tweets import split_for_id

SEED = 13
TARGET_COLUMN = 'market_moving'


@dataclass(frozen=True)
class Dataset:
    train_records: list[FeatureRecord]
    train_labels: list[int]
    dev_records: list[FeatureRecord]
    dev_labels: list[int]


def load_dataset(csv_path: Path) -> Dataset:
    buckets: dict[str, tuple[list[FeatureRecord], list[int]]] = {
        'train': ([], []),
        'dev': ([], []),
        'test': ([], []),
    }
    with csv_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            records, labels = buckets[split_for_id(row['id'])]
            records.append(FeatureRecord(text=row['text'], source=row['source']))
            labels.append(int(row[TARGET_COLUMN]))
    train, dev = buckets['train'], buckets['dev']
    return Dataset(train[0], train[1], dev[0], dev[1])


def _candidates() -> dict[str, Any]:
    models: dict[str, Any] = {
        'logreg': LogisticRegression(max_iter=1000, C=4.0, random_state=SEED),
    }
    if importlib.util.find_spec('xgboost') is not None:
        from xgboost import XGBClassifier

        models['xgboost'] = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.2, random_state=SEED, n_jobs=1
        )
    return models


def train_tier1(csv_path: Path, out_dir: Path) -> dict[str, dict[str, float]]:
    data = load_dataset(csv_path)
    if len(set(data.dev_labels)) < 2:
        raise ValueError('dev split needs both classes to compute auc')

    featurizer = TweetFeaturizer().fit(data.train_records)
    train_x = featurizer.transform(data.train_records)
    dev_x = featurizer.transform(data.dev_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for name, model in _candidates().items():
        model.fit(train_x, data.train_labels)
        scores = model.predict_proba(dev_x)[:, 1]
        metrics[name] = {
            'auc': float(roc_auc_score(data.dev_labels, scores)),
            'accuracy': float(accuracy_score(data.dev_labels, scores >= 0.5)),
            'dev_rows': float(len(data.dev_labels)),
            'train_rows': float(len(data.train_labels)),
        }
        joblib.dump({'featurizer': featurizer, 'model': model}, out_dir / f'tier1_{name}.joblib')

    (out_dir / 'tier1_metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the tier one tweet baselines')
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(train_tier1(args.csv_path, args.out_dir), indent=2))


if __name__ == '__main__':
    main()
