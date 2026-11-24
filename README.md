# model-foundry

I train and ship the cheaper rungs of the model ladder. This repo is where a task leaves the paid frontier tier and lands on something small, fast and measured.

The thesis is simple. Never send work to an expensive model that a cheaper rung handles. Rules first, then classical ML, then small encoders, then fine tuned small LLMs, then paid frontier models.

## The ladder

<!-- ladder:start -->
| Rung | Tier | Task | Model | Owner repo | Latency | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Rules | Obvious feed filtering and market keyword routing | Regex and keyword tables | the consuming service | Microseconds, in process | Free |
| 2 | Classical ML | Tweet market impact baseline | TF-IDF plus engineered features, logistic regression or xgboost | model-foundry | About a millisecond on CPU | Free after training, trains in seconds |
| 3 | Small encoder | Prompt injection detection on inbound text | DeBERTa v3 small with a classifier head | model-foundry | Tens of milliseconds on one GPU or CPU | Fixed instance cost, no per token fee |
| 4 | Fine tuned small LLM | Tweet classification, bet brief drafting, intent routing | Qwen2.5 1.5B, Llama 3.1 8B and Gemma 2 2B with LoRA adapters | model-foundry | Low hundreds of milliseconds on vLLM, one host per base model | Small GPUs per base, adapters keep per task cost low |
| 5 | Frontier | Open ended reasoning and the hard residual | Paid frontier models | the agent services | Seconds | Per token, highest |
<!-- ladder:end -->

The table is generated from `foundry/ladder.py`. Run `python scripts/render_ladder.py` after changing it. The latency and cost cells are planning notes, not benchmarks. Real numbers come from llm-ops once a rung is live.

## Rung 1, rules

Rules are not trained here. They live in the services that consume the feed. I list them so every other rung has a floor to beat. If a regex does the job, nothing below it in this repo should exist.

## Rung 2, classical ML

`foundry/baselines` builds a TF-IDF matrix and adds three engineered signals. A count of injury words, the share of capital letters, and a prior on how reliable the source is. Logistic regression always trains. Xgboost trains too when it is installed. It runs in seconds on the fixture and writes a joblib file plus a metrics json.

This rung answers one question. Is this post likely to move a market at all. Most posts are not. A frontier model should never see them.

```
python -m foundry.baselines.train_tier1 data/fixtures/tweets_fixture.csv artifacts/tier1
```

## Rung 3, small encoder

`foundry/encoders/train_injection.py` fine tunes DeBERTa v3 small as a prompt injection detector. It is config driven, stops early on dev F1 and exports safetensors. It needs the train extra.

Every agent that reads untrusted text sits behind this guard. A guard prompt to a frontier model costs a full call per message. The encoder costs a fixed instance.

## Rung 4, fine tuned small LLMs

Three tasks, three configs in `foundry/finetune/configs`.

- Tweet classifier on Qwen2.5 1.5B Instruct with 4 bit QLoRA, rank 16, alpha 32, three epochs, cosine schedule.
- Bet drafter on Llama 3.1 8B Instruct, also 4 bit QLoRA, trained on settled bets and closing line moves.
- Intent router on Gemma 2 2B Instruct with LoRA.

One entry point runs all of them. It reads a yaml config, validates every row against the task schema, formats chats through `foundry/finetune/prompts.py`, pins seeds and writes safetensors only.

```
pip install -e '.[train]'
python -m foundry.finetune.sft foundry/finetune/configs/tweet_classifier_qwen25_15b_qlora.yaml
```

Serving uses vLLM with one host per base model and the task adapter loaded onto it. The Qwen host config in `foundry/serving` is the worked example. `scripts/smoke_vllm.py` checks a live server and is never run in CI.

## Labels and data

A frontier teacher labels raw posts and a seeded sample goes to a human audit. `foundry/labeling/teacher.py` asks for strict JSON and rejects anything that does not match the schema. The clients are injected, so tests use a fake.

`foundry/datasets/build_tweets.py` turns a trace export into labeled rows. It drops near duplicates with banded minhash, strips emails, phone numbers, links and handles, and assigns train, dev and test by a hash of the id so a row never changes split.

The golden set is frozen with a sha256 manifest. `golden.load` refuses a file that drifted. A small sample is committed in `data/golden`. The full set stays out of the repo.

All fixtures are synthetic. Players and teams are invented.

## What each rung replaced

- Rung 2 replaced frontier triage calls on posts that were routine or irrelevant. A millisecond on CPU beats a seconds long paid call, and most of the feed is that kind of post.
- Rung 3 replaced guard prompts that asked a frontier model whether a message was an injection. The task is a yes or no on short text. An encoder does it in tens of milliseconds with no per token fee.
- Rung 4 replaced frontier calls whose output has a fixed shape. Classification into a JSON object, a short brief from a signal, and routing to one intent. A small tuned model is cheaper per call and faster, and each base model runs on a small vLLM host so the GPU bill stays flat while call volume grows.

The frontier tier stays for open ended reasoning and the cases a cheaper rung abstains on.

## The release rule

No rung ships without passing evalgate. Training metrics in a model card are not a gate. evalgate runs the frozen golden set and decides. llm-ops then watches the rung live and flags drift.

`foundry/hub/push.py` creates the Hub repo if it is missing and uploads the card, the train config and the adapter. It reads `HF_TOKEN` from the environment and refuses `.bin`, `.pt` and other pickle formats. Cards are rendered from templates in `foundry/hub/cards` and refuse to render with missing fields.

## Setup

The shared types come from sportsbook-contracts. Install it first.

```
pip install -e ../sportsbook-contracts/packages/py
pip install -e '.[dev]'
ruff check .
pytest
```

Extras are `train`, `label` and `serve`. Tests never need them, never download a model and never touch the network. Copy `.env.example` for the keys that the labeling, serving and push scripts read.

## TODO

The injection detector has a config and a trainer but no committed fixture data yet.

## License

MIT
