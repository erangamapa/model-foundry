"""Minimal OpenAI compatible chat client for a vLLM server. Standard library only."""

import json
import os
import urllib.request
from collections.abc import Callable

Transport = Callable[[str, dict[str, str], bytes], bytes]

REQUEST_TIMEOUT_SECONDS = 30


class MissingBaseUrlError(Exception):
    pass


def _http_post(url: str, headers: dict[str, str], body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


class VllmClient:
    """Satisfies the teacher client protocol, so a student can be swapped in for labeling."""

    def __init__(self, base_url: str, model: str, transport: Transport = _http_post) -> None:
        self._endpoint = base_url.rstrip('/') + '/chat/completions'
        self._model = model
        self._transport = transport

    @classmethod
    def from_env(cls, model: str) -> 'VllmClient':
        base_url = os.environ.get('VLLM_BASE_URL')
        if not base_url:
            raise MissingBaseUrlError('VLLM_BASE_URL is not set')
        return cls(base_url, model)

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        payload = {
            'model': self._model,
            'temperature': 0,
            'max_tokens': max_tokens,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        }
        raw = self._transport(
            self._endpoint,
            {'Content-Type': 'application/json'},
            json.dumps(payload).encode(),
        )
        return json.loads(raw)['choices'][0]['message']['content']
