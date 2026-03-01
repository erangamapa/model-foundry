import json

import pytest

from foundry.finetune.prompts import (
    SYSTEMS,
    prompt_messages,
    training_messages,
    validate_row,
)

TWEET = {
    'id': 't1', 'text': 'Kofi Merriweather out for six weeks.', 'source': 'wire_desk',
    'event_type': 'injury', 'entity': 'Kofi Merriweather', 'certainty': 'confirmed',
    'market_direction': 'down', 'label_source': 'human',
}
DRAFT = {
    'id': 'd1',
    'signal': {'pick': 'Brightwater', 'price': '2.10'},
    'brief': 'Back Brightwater at 2.10.',
}
INTENT = {'id': 'i1', 'text': 'how much do I have left', 'intent': 'check_balance'}


def test_no_empty_system_prompt() -> None:
    assert set(SYSTEMS) == {'tweet_classifier', 'drafter', 'intent_router'}
    assert all(len(text.strip()) > 40 for text in SYSTEMS.values())


@pytest.mark.parametrize(
    ('task', 'row'),
    [('tweet_classifier', TWEET), ('drafter', DRAFT), ('intent_router', INTENT)],
)
def test_training_messages_shape(task: str, row: dict) -> None:
    messages = training_messages(task, validate_row(task, row))  # type: ignore[arg-type]
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
    assert all(m['content'] for m in messages)


def test_tweet_target_is_strict_json_of_labels() -> None:
    target = training_messages('tweet_classifier', TWEET)[-1]['content']
    assert json.loads(target) == {
        'event_type': 'injury', 'entity': 'Kofi Merriweather',
        'certainty': 'confirmed', 'market_direction': 'down',
    }


def test_drafter_user_turn_is_canonical_signal_json() -> None:
    user = prompt_messages('drafter', DRAFT)[1]['content']
    assert user == '{"pick":"Brightwater","price":"2.10"}'


def test_prompt_messages_have_no_answer() -> None:
    assert [m['role'] for m in prompt_messages('intent_router', INTENT)] == ['system', 'user']


def test_invalid_row_is_rejected_before_training() -> None:
    with pytest.raises(ValueError):
        validate_row('intent_router', {**INTENT, 'intent': 'dance'})


def test_gemma_folds_system_into_user() -> None:
    row = validate_row('intent_router', INTENT)
    messages = prompt_messages('intent_router', row, 'google/gemma-2-2b-it')
    assert [m['role'] for m in messages] == ['user']
    assert SYSTEMS['intent_router'] in messages[0]['content']
    assert INTENT['text'] in messages[0]['content']


def test_system_role_kept_for_models_that_support_it() -> None:
    row = validate_row('tweet_classifier', TWEET)
    messages = prompt_messages('tweet_classifier', row, 'Qwen/Qwen2.5-1.5B-Instruct')
    assert [m['role'] for m in messages] == ['system', 'user']
