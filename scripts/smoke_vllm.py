"""Smoke test against a live vLLM server. Needs VLLM_BASE_URL, not run in CI."""

import json
import os
import sys

from foundry.finetune.prompts import system_prompt, user_content
from foundry.serving.client import MissingBaseUrlError, VllmClient

ADAPTER = os.environ.get('VLLM_ADAPTER', 'tweet-classifier')
SAMPLE = {'text': 'Harbor City midfielder Dario Ketteridge ruled out with a hamstring injury.'}


def main() -> int:
    try:
        client = VllmClient.from_env(ADAPTER)
    except MissingBaseUrlError as error:
        print(f'skipped, {error}')
        return 0
    reply = client.complete(
        system_prompt('tweet_classifier'), user_content('tweet_classifier', SAMPLE)
    )
    print(reply)
    json.loads(reply)
    return 0


if __name__ == '__main__':
    sys.exit(main())
