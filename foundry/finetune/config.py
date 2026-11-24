from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Task = Literal['tweet_classifier', 'drafter', 'intent_router']


class ConfigError(Exception):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class QuantConfig(Strict):
    load_in_4bit: bool = True
    quant_type: Literal['nf4', 'fp4'] = 'nf4'
    compute_dtype: Literal['bfloat16', 'float16'] = 'bfloat16'
    double_quant: bool = True


class LoraSettings(Strict):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class TrainingSettings(Strict):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lr_scheduler: Literal['cosine', 'linear', 'constant']
    warmup_ratio: float = Field(ge=0, lt=1)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    packing: bool = False
    weight_decay: float = Field(default=0.0, ge=0)


class DataSettings(Strict):
    train_path: Path
    dev_path: Path


class TrainConfig(Strict):
    name: str = Field(min_length=1)
    task: Task
    base_model: str = Field(min_length=3)
    seed: int
    output_dir: Path
    data: DataSettings
    quantization: QuantConfig | None = None
    lora: LoraSettings
    training: TrainingSettings

    @model_validator(mode='after')
    def _alpha_not_below_rank(self) -> 'TrainConfig':
        if self.lora.alpha < self.lora.r:
            raise ValueError('lora alpha must be at least the rank')
        return self


def load_config(path: Path) -> TrainConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
        return TrainConfig.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f'{path.name} is not a valid train config. {error}') from error
