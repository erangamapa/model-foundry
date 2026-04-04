"""DeBERTa v3 small injection classifier. Needs the train extra, imports are lazy."""

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from foundry.hub.push import assert_safetensors_only


class InjectionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    base_model: str = 'microsoft/deberta-v3-small'
    train_path: Path
    dev_path: Path
    output_dir: Path
    seed: int = 13
    max_length: int = Field(default=256, gt=0)
    epochs: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    early_stopping_patience: int = Field(default=2, gt=0)


def load_config(path: Path) -> InjectionConfig:
    return InjectionConfig.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))


def read_rows(path: Path) -> list[dict]:
    """Rows are {"text": str, "label": 0 or 1}. One means injection."""
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(config: InjectionConfig) -> Path:
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=2, use_safetensors=True
    )

    def encode(rows: list[dict]) -> Dataset:
        data = Dataset.from_list(rows)
        return data.map(
            lambda batch: tokenizer(
                batch['text'], truncation=True, max_length=config.max_length
            ),
            batched=True,
        )

    def compute_metrics(prediction: tuple) -> dict[str, float]:
        logits, labels = prediction
        return {'f1': float(f1_score(labels, np.argmax(logits, axis=-1)))}

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        save_safetensors=True,
        seed=config.seed,
        data_seed=config.seed,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encode(read_rows(config.train_path)),
        eval_dataset=encode(read_rows(config.dev_path)),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()

    export_dir = config.output_dir / 'export'
    trainer.model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)
    assert_safetensors_only(sorted(export_dir.rglob('*')))
    (export_dir / 'metrics.json').write_text(
        json.dumps(trainer.evaluate(), indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    return export_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the injection detector')
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == '__main__':
    main()
