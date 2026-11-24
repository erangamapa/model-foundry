from pathlib import Path

import pytest

from foundry.ladder import LADDER, render_table
from foundry.serving.client import MissingBaseUrlError, VllmClient
from scripts.render_ladder import END, START, replace_between_markers


def test_ladder_is_ordered_and_complete() -> None:
    assert [r.rung for r in LADDER] == [1, 2, 3, 4, 5]
    assert all(r.replaces and r.latency_note and r.cost_note for r in LADDER)


def test_table_has_a_row_per_rung() -> None:
    assert len(render_table().splitlines()) == len(LADDER) + 2


def test_marker_replacement_is_idempotent() -> None:
    text = f'intro\n{START}\nold\n{END}\noutro\n'
    once = replace_between_markers(text, 'TABLE')
    assert once == f'intro\n{START}\nTABLE\n{END}\noutro\n'
    assert replace_between_markers(once, 'TABLE') == once


def test_missing_markers_fail() -> None:
    with pytest.raises(ValueError):
        replace_between_markers('no markers', 'TABLE')


def test_readme_table_is_current(root: Path) -> None:
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert replace_between_markers(readme, render_table()) == readme


def test_vllm_client_uses_injected_transport() -> None:
    seen: dict = {}

    def transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
        seen['url'] = url
        seen['body'] = body
        return b'{"choices":[{"message":{"content":"ok"}}]}'

    client = VllmClient('http://vllm.sportsbook.internal:8000/v1/', 'tweet-classifier', transport)
    assert client.complete('sys', 'user') == 'ok'
    assert seen['url'] == 'http://vllm.sportsbook.internal:8000/v1/chat/completions'
    assert b'tweet-classifier' in seen['body']


def test_vllm_client_needs_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('VLLM_BASE_URL', raising=False)
    with pytest.raises(MissingBaseUrlError):
        VllmClient.from_env('m')
