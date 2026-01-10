import pytest
from pydantic import ValidationError

from foundry.datasets.schemas import DrafterExample, IntentExample, LabeledTweet

VALID = {
    'id': 't1',
    'text': 'Dario Ketteridge ruled out.',
    'source': 'wire_desk',
    'event_type': 'injury',
    'entity': 'Dario Ketteridge',
    'certainty': 'confirmed',
    'market_direction': 'down',
    'label_source': 'human',
}


def test_valid_tweet_parses() -> None:
    assert LabeledTweet.model_validate(VALID).event_type == 'injury'


@pytest.mark.parametrize(
    'patch',
    [
        {'event_type': 'earthquake'},
        {'label_source': 'crowd'},
        {'text': ''},
        {'extra_field': 1},
        {'certainty': 3},
    ],
)
def test_bad_tweet_rejected(patch: dict) -> None:
    with pytest.raises(ValidationError):
        LabeledTweet.model_validate({**VALID, **patch})


def test_tweet_is_frozen() -> None:
    tweet = LabeledTweet.model_validate(VALID)
    with pytest.raises(ValidationError):
        tweet.text = 'changed'  # type: ignore[misc]


def test_drafter_and_intent_examples() -> None:
    assert DrafterExample.model_validate({'id': 'd1', 'signal': {'pick': 'x'}, 'brief': 'b'})
    intent = {'id': 'i1', 'text': 'what is my balance', 'intent': 'check_balance'}
    assert IntentExample.model_validate(intent)
    with pytest.raises(ValidationError):
        IntentExample.model_validate({'id': 'i1', 'text': 'hi', 'intent': 'dance'})
