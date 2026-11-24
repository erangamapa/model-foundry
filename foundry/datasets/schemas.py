from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal['injury', 'lineup', 'trade', 'suspension', 'result', 'rumor', 'other']
Certainty = Literal['confirmed', 'likely', 'rumor']
MarketDirection = Literal['up', 'down', 'neutral']
LabelSource = Literal['teacher', 'human']
Intent = Literal['place_bet', 'check_balance', 'odds_question', 'bet_status', 'set_limit', 'other']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class LabeledTweet(Strict):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)
    source: str = Field(min_length=1)
    event_type: EventType
    entity: str = Field(min_length=1)
    certainty: Certainty
    market_direction: MarketDirection
    label_source: LabelSource


class DrafterExample(Strict):
    id: str = Field(min_length=1)
    signal: dict[str, Any]
    brief: str = Field(min_length=1)


class IntentExample(Strict):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=1000)
    intent: Intent
