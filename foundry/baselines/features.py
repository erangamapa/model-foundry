from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

INJURY_WORDS = frozenset(
    {
        'injury', 'injured', 'hamstring', 'ankle', 'knee', 'concussion', 'sidelined',
        'strain', 'sprain', 'doubtful', 'questionable', 'out', 'surgery', 'limped',
    }
)

DEFAULT_RELIABILITY = 0.5
SOURCE_RELIABILITY: dict[str, float] = {
    'club_official': 0.95,
    'wire_desk': 0.85,
    'beat_reporter': 0.7,
    'fan_account': 0.2,
    'aggregator': 0.35,
}

FEATURE_NAMES = ('injury_word_count', 'caps_ratio', 'source_reliability')


@dataclass(frozen=True)
class FeatureRecord:
    text: str
    source: str


def injury_word_count(text: str) -> int:
    tokens = (token.strip('.,!?:;()"\'').lower() for token in text.split())
    return sum(token in INJURY_WORDS for token in tokens)


def caps_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def source_reliability(source: str) -> float:
    return SOURCE_RELIABILITY.get(source, DEFAULT_RELIABILITY)


def engineered(records: Sequence[FeatureRecord]) -> csr_matrix:
    rows = [
        [injury_word_count(r.text), caps_ratio(r.text), source_reliability(r.source)]
        for r in records
    ]
    return csr_matrix(np.asarray(rows, dtype=float))


class TweetFeaturizer:
    """Word and bigram tfidf stacked with three hand built signals."""

    def __init__(self, max_features: int = 2000) -> None:
        self._tfidf = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_features=max_features
        )

    def fit(self, records: Sequence[FeatureRecord]) -> 'TweetFeaturizer':
        self._tfidf.fit([r.text for r in records])
        return self

    def transform(self, records: Sequence[FeatureRecord]) -> csr_matrix:
        text_part = self._tfidf.transform([r.text for r in records])
        return hstack([text_part, engineered(records)], format='csr')

    def fit_transform(self, records: Sequence[FeatureRecord]) -> csr_matrix:
        return self.fit(records).transform(records)
