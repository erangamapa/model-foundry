from pathlib import Path

import pytest
import yaml

from foundry.encoders.train_injection import load_config as load_injection
from foundry.finetune.config import ConfigError, TrainConfig, load_config
from foundry.serving.config import VllmServeConfig, load_serve_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'foundry' / 'finetune' / 'configs'
SERVING_DIR = Path(__file__).resolve().parent.parent / 'foundry' / 'serving'


@pytest.mark.parametrize('path', sorted(CONFIG_DIR.glob('*.yaml')), ids=lambda p: p.name)
def test_every_train_config_parses(path: Path) -> None:
    config = load_config(path)
    assert isinstance(config, TrainConfig)
    assert config.training.packing is False


def test_tweet_recipe_matches_the_brief() -> None:
    config = load_config(CONFIG_DIR / 'tweet_classifier_qwen25_15b_qlora.yaml')
    assert config.base_model == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert config.quantization is not None and config.quantization.quant_type == 'nf4'
    assert (config.lora.r, config.lora.alpha) == (16, 32)
    assert config.lora.target_modules == ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    assert config.training.epochs == 3
    assert config.training.lr_scheduler == 'cosine'


def _write(tmp_path: Path, patch: dict) -> Path:
    raw = yaml.safe_load((CONFIG_DIR / 'intent_router_gemma2_2b_lora.yaml').read_text())
    for key, value in patch.items():
        section, _, field = key.partition('.')
        if field:
            raw[section][field] = value
        else:
            raw[section] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    'patch',
    [
        {'task': 'poetry'},
        {'lora.r': 0},
        {'lora.r': 32, 'lora.alpha': 8},
        {'training.lr_scheduler': 'wobbly'},
        {'surprise': True},
    ],
)
def test_bad_config_rejected(tmp_path: Path, patch: dict) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, patch))


def test_serving_configs_parse_and_render_args() -> None:
    for name in ('vllm_shared.yaml', 'vllm_tweet_classifier.yaml'):
        config = load_serve_config(SERVING_DIR / name)
        args = config.to_args()
        assert args[:3] == ['vllm', 'serve', config.model]
        assert '--enable-lora' in args


def test_serving_rejects_more_adapters_than_slots() -> None:
    with pytest.raises(ValueError):
        VllmServeConfig(
            model='m', served_model_name='m', gpu_memory_utilization=0.5, max_model_len=10,
            port=8000, max_loras=1, lora_modules={'a': 'x', 'b': 'y'},
        )


def test_injection_config_parses() -> None:
    config = load_injection(
        Path(__file__).resolve().parent.parent
        / 'foundry' / 'encoders' / 'configs' / 'injection_deberta_v3_small.yaml'
    )
    assert config.base_model == 'microsoft/deberta-v3-small'
