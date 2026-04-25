from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

TEMPLATE_DIR = Path(__file__).parent / 'cards'
CardTask = Literal['tweet_classifier', 'drafter', 'intent_router', 'injection_detector']

# adapters carry peft metadata, the encoder is a full fine tune served with transformers
LIBRARY_BY_TASK: dict[str, str] = {
    'tweet_classifier': 'peft',
    'drafter': 'peft',
    'intent_router': 'peft',
    'injection_detector': 'transformers',
}


class CardError(Exception):
    pass


class CardContext(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model_id: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    training_rows: int = Field(gt=0)
    dev_rows: int = Field(gt=0)
    seed: int
    metrics: dict[str, float] = Field(min_length=1)


_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


def render_card(task: CardTask, context: dict) -> str:
    try:
        validated = CardContext.model_validate(context)
    except ValidationError as error:
        raise CardError(f'card context is incomplete. {error}') from error
    try:
        return _env.get_template(f'{task}.md.j2').render(
            **validated.model_dump(), library_name=LIBRARY_BY_TASK[task]
        )
    except TemplateError as error:
        raise CardError(str(error)) from error
