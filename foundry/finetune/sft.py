"""One SFT entry point for every task config. Needs the train extra, imports are lazy."""

import argparse
import json
import random
from pathlib import Path

from foundry.finetune.config import TrainConfig, load_config
from foundry.finetune.prompts import training_messages, validate_row
from foundry.hub.push import assert_safetensors_only


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_chat_rows(config: TrainConfig, path: Path) -> list[dict]:
    return [
        {
            'messages': training_messages(
                config.task, validate_row(config.task, row), config.base_model
            )
        }
        for row in read_jsonl(path)
    ]


def pin_seeds(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def run(config: TrainConfig) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    pin_seeds(config.seed)

    compute_dtype = torch.bfloat16
    if config.quantization is not None:
        compute_dtype = getattr(torch, config.quantization.compute_dtype)
    quantization = None
    if config.quantization is not None:
        quantization = BitsAndBytesConfig(
            load_in_4bit=config.quantization.load_in_4bit,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.compute_dtype),
            bnb_4bit_use_double_quant=config.quantization.double_quant,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        use_safetensors=True,
    )

    train = Dataset.from_list(build_chat_rows(config, config.data.train_path))
    dev = Dataset.from_list(build_chat_rows(config, config.data.dev_path))
    settings = config.training

    args = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_length=settings.max_seq_length,
        packing=settings.packing,
        seed=config.seed,
        data_seed=config.seed,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_safetensors=True,
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        report_to='none',
    )
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type='CAUSAL_LM',
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = config.output_dir / 'adapter'
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    assert_safetensors_only(sorted(adapter_dir.rglob('*')))
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SFT from a yaml train config')
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == '__main__':
    main()
