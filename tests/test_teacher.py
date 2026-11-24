import json

import pytest

from foundry.datasets.schemas import LabeledTweet
from foundry.labeling.audit import pick_audit_sample
from foundry.labeling.teacher import (
    MissingCredentialError,
    RawItem,
    anthropic_teacher_from_env,
    label_batch,
    openai_teacher_from_env,
    parse_label,
)

GOOD = json.dumps(
    {
        'event_type': 'injury',
        'entity': 'Misha Oranov',
        'certainty': 'likely',
        'market_direction': 'down',
    }
)


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return next(self._replies)


def _items(n: int) -> list[RawItem]:
    return [RawItem(id=f'i{k}', text=f'post number {k}', source='wire_desk') for k in range(n)]


def test_valid_reply_becomes_teacher_labeled_tweet() -> None:
    client = FakeClient([f'Here you go\n{GOOD}'])
    result = label_batch(_items(1), client)
    assert result.rejected == []
    assert result.labeled[0].label_source == 'teacher'
    assert result.labeled[0].entity == 'Misha Oranov'
    assert 'post number 0' in client.calls[0][1]


@pytest.mark.parametrize(
    'reply',
    [
        'not json at all',
        '{"event_type": "injury"}',
        GOOD.replace('injury', 'meteor'),
        GOOD.replace('Misha Oranov', ''),
        GOOD[:-1],
    ],
)
def test_malformed_reply_is_rejected(reply: str) -> None:
    result = label_batch(_items(1), FakeClient([reply]))
    assert result.labeled == []
    assert result.rejected[0].item_id == 'i0'


def test_one_bad_reply_does_not_sink_the_batch() -> None:
    result = label_batch(_items(3), FakeClient([GOOD, 'nope', GOOD]))
    assert [t.id for t in result.labeled] == ['i0', 'i2']
    assert [r.item_id for r in result.rejected] == ['i1']


def test_parse_label_rejects_extra_keys() -> None:
    with pytest.raises(ValueError):
        parse_label(GOOD[:-1] + ', "confidence": 0.9}')


def test_factories_need_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(MissingCredentialError):
        anthropic_teacher_from_env()
    with pytest.raises(MissingCredentialError):
        openai_teacher_from_env()


def _batch(n: int) -> list[LabeledTweet]:
    return [
        LabeledTweet(
            id=f'r{k}', text='x', source='s', event_type='other', entity='e',
            certainty='rumor', market_direction='neutral', label_source='teacher',
        )
        for k in range(n)
    ]


def test_audit_sample_is_deterministic_and_order_independent() -> None:
    batch = _batch(50)
    first = pick_audit_sample(batch, 0.1, seed=3)
    assert len(first) == 5
    assert first == pick_audit_sample(batch, 0.1, seed=3)
    assert first == pick_audit_sample(list(reversed(batch)), 0.1, seed=3)
    assert first != pick_audit_sample(batch, 0.1, seed=4)


def test_audit_sample_edges() -> None:
    assert pick_audit_sample([], 0.5, seed=1) == []
    assert len(pick_audit_sample(_batch(3), 0.01, seed=1)) == 1
    with pytest.raises(ValueError):
        pick_audit_sample(_batch(3), 0, seed=1)
