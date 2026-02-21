import json
from pathlib import Path

import joblib

from foundry.baselines.features import (
    FeatureRecord,
    TweetFeaturizer,
    caps_ratio,
    injury_word_count,
    source_reliability,
)
from foundry.baselines.train_tier1 import train_tier1


def test_engineered_features() -> None:
    assert injury_word_count('Ankle strain, doubtful for Saturday.') == 3
    assert injury_word_count('Great win today') == 0
    assert caps_ratio('ABcd') == 0.5
    assert caps_ratio('1234') == 0.0
    assert source_reliability('club_official') > source_reliability('fan_account')
    assert source_reliability('unknown_source') == 0.5


def test_featurizer_shape() -> None:
    records = [FeatureRecord('Knee injury', 'wire_desk'), FeatureRecord('Nice day', 'fan_account')]
    matrix = TweetFeaturizer().fit_transform(records)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 3


def test_tier1_trains_on_fixture(fixtures_dir: Path, tmp_path: Path) -> None:
    metrics = train_tier1(fixtures_dir / 'tweets_fixture.csv', tmp_path)
    assert metrics['logreg']['auc'] > 0.7
    written = json.loads((tmp_path / 'tier1_metrics.json').read_text())
    assert written == metrics
    bundle = joblib.load(tmp_path / 'tier1_logreg.joblib')
    record = FeatureRecord('Hamstring injury rules him out', 'club_official')
    probe = bundle['featurizer'].transform([record])
    assert bundle['model'].predict_proba(probe)[0, 1] > 0.5
