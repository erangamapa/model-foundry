import pytest

from foundry.hub.card import CardError, render_card

CONTEXT = {
    'model_id': 'oddsmith/tweet-classifier-qwen25-15b',
    'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
    'training_rows': 1200,
    'dev_rows': 150,
    'seed': 13,
    'metrics': {'macro_f1': 0.8731, 'accuracy': 0.91},
}


@pytest.mark.parametrize(
    'task', ['tweet_classifier', 'drafter', 'intent_router', 'injection_detector']
)
def test_every_card_renders_with_metrics(task: str) -> None:
    card = render_card(task, CONTEXT)  # type: ignore[arg-type]
    assert '| macro_f1 | 0.873 |' in card
    assert 'base_model: Qwen/Qwen2.5-1.5B-Instruct' in card
    assert 'safetensors' in card
    assert '## Bias and limits' in card
    assert '{{' not in card and '{%' not in card


@pytest.mark.parametrize('missing', list(CONTEXT))
def test_missing_field_is_refused(missing: str) -> None:
    context = {k: v for k, v in CONTEXT.items() if k != missing}
    with pytest.raises(CardError):
        render_card('tweet_classifier', context)


def test_empty_metrics_refused() -> None:
    with pytest.raises(CardError):
        render_card('drafter', {**CONTEXT, 'metrics': {}})
