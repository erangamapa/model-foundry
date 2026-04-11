from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VllmServeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    model: str
    served_model_name: str
    dtype: str = 'bfloat16'
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(gt=0)
    port: int = Field(gt=0, lt=65536)
    enable_lora: bool = True
    max_loras: int = Field(default=1, gt=0)
    max_lora_rank: int = Field(default=16, gt=0)
    lora_modules: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _adapters_fit_slots(self) -> 'VllmServeConfig':
        if len(self.lora_modules) > self.max_loras:
            raise ValueError('more lora modules than max_loras slots')
        if self.lora_modules and not self.enable_lora:
            raise ValueError('lora modules need enable_lora')
        return self

    def to_args(self) -> list[str]:
        args = [
            'vllm', 'serve', self.model,
            '--served-model-name', self.served_model_name,
            '--dtype', self.dtype,
            '--gpu-memory-utilization', str(self.gpu_memory_utilization),
            '--max-model-len', str(self.max_model_len),
            '--port', str(self.port),
        ]
        if self.enable_lora:
            args += ['--enable-lora', '--max-loras', str(self.max_loras)]
            args += ['--max-lora-rank', str(self.max_lora_rank)]
            args += ['--lora-modules', *(f'{n}={p}' for n, p in self.lora_modules.items())]
        return args


def load_serve_config(path: Path) -> VllmServeConfig:
    return VllmServeConfig.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
