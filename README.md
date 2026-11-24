# model-foundry

Most LLM workloads send everything to the biggest model. This repo is a ladder of cheaper models for the tasks that do not need one, with the training code, the data pipeline and the release rule for each rung. I built it for a sportsbook platform, so the worked example is tweet triage, prompt injection detection and bet briefs. I train and ship the cheaper rungs of the model ladder. This repo is where a task leaves the paid frontier tier and lands on something small, fast and measured.

The thesis is simple. Never send work to an expensive model that a cheaper rung handles. Rules first, then classical ML, then small encoders, then fine tuned small LLMs, then paid frontier models.

## The ladder

<!-- ladder:start -->
| Rung | Tier | Task | Model | Owner repo | Latency | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Rules | Obvious feed filtering and market keyword routing | Regex and keyword tables | the consuming service | Microseconds, in process | Free |
| 2 | Classical ML | Tweet market impact baseline | TF-IDF plus engineered features, logistic regression or xgboost | model-foundry | About a millisecond on CPU | Free after training, trains in seconds |
| 3 | Small encoder | Prompt injection detection on inbound text | DeBERTa v3 small with a classifier head | model-foundry | Tens of milliseconds on one GPU or CPU | Fixed instance cost, no per token fee |
| 4 | Fine tuned small LLM | Tweet classification, bet brief drafting, intent routing | Qwen2.5 1.5B and Gemma 2 2B with LoRA adapters, Llama 3.1 8B config only | model-foundry | Low hundreds of milliseconds on vLLM, one host per base model | Small GPUs per base, adapters keep per task cost low |
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

The intended use is a cheap first-layer check on untrusted text. A guard prompt to a frontier model costs a full call per message, while the encoder costs a fixed instance. I have not measured that cost difference. The measured results are in the section below.

## Rung 4, fine tuned small LLMs

Three tasks, three configs in `foundry/finetune/configs`.

- Tweet classifier on Qwen2.5 1.5B Instruct with plain LoRA (no quantisation), rank 16, alpha 32, five epochs, cosine schedule. Trained.
- Intent router on Gemma 2 2B Instruct with LoRA, rank 8, five epochs. Trained.
- Bet drafter on Llama 3.1 8B Instruct with 4 bit QLoRA. This is a config only. It has not been trained and no weights exist.

One entry point runs all of them. It reads a yaml config, validates every row against the task schema, formats chats through `foundry/finetune/prompts.py`, pins seeds and writes safetensors only.

```
python -m foundry.finetune.sft foundry/finetune/configs/tweet_classifier_qwen25_15b_lora.yaml
```

Serving uses vLLM with one host per base model and the task adapter loaded onto it. The Qwen host config in `foundry/serving` is the worked example. `scripts/smoke_vllm.py` checks a live server and is never run in CI.

## Labels and data

A frontier teacher labels raw posts and a seeded sample goes to a human audit. `foundry/labeling/teacher.py` asks for strict JSON and rejects anything that does not match the schema. The clients are injected, so tests use a fake.

`foundry/datasets/build_tweets.py` turns a trace export into labeled rows. It drops near duplicates with banded minhash, strips emails, phone numbers, links and handles, and assigns train, dev and test by a hash of the id so a row never changes split.

The golden set is frozen with a sha256 manifest. `golden.load` refuses a file that drifted. A small sample is committed in `data/golden`. The full set stays out of the repo.

All fixtures are synthetic. Players and teams are invented.

## What each rung is meant to replace

This is the intent. I have not measured cost or latency savings on live traffic, and none of these models serves production traffic.

- Rung 2 is meant to replace frontier triage calls on posts that were routine or irrelevant. A millisecond on CPU beats a seconds long paid call, and most of the feed is that kind of post.
- Rung 3 is meant to replace guard prompts that asked a frontier model whether a message was an injection. The task is a yes or no on short text. An encoder does it in tens of milliseconds with no per token fee.
- Rung 4 is meant to replace frontier calls whose output has a fixed shape. Classification into a JSON object, a short brief from a signal, and routing to one intent. A small tuned model is cheaper per call and faster, and each base model runs on a small vLLM host so the GPU bill stays flat while call volume grows.

The frontier tier stays for open ended reasoning and the cases a cheaper rung abstains on.

## Results

Three small experiments are trained and published on the Hub, under [huggingface.co/erangamapa](https://huggingface.co/erangamapa). All numbers are measured on held-out test splits, trained on an Apple M1 Max, and the raw outputs are in each repo's `eval_results.json`. None of these models serves production traffic, and evalgate was not run on them. These are small-data experiments, not production claims.

| Model | Task | Data | Test result | Baselines |
| --- | --- | --- | --- | --- |
| [injection-detector-deberta-v3-small](https://huggingface.co/erangamapa/injection-detector-deberta-v3-small) | Prompt injection, binary | `deepset/prompt-injections`, 480 train, 66 dev, its own 116 row test split | F1 0.966, accuracy 0.966, precision 0.983, recall 0.950 | TF-IDF plus logistic regression F1 0.763, majority class F1 0.000 |
| [tweet-outcome-classifier-qwen2.5-1.5b-lora](https://huggingface.co/erangamapa/tweet-outcome-classifier-qwen2.5-1.5b-lora) | Four field tweet label as JSON | the synthetic fixture in this repo, 47 train, 10 dev, the 30 row golden sample as test | Exact match on all four fields 0.90 | TF-IDF plus logistic regression matches or beats it on the three categorical fields but cannot predict `entity` |
| [intent-router-gemma-2-2b-lora](https://huggingface.co/erangamapa/intent-router-gemma-2-2b-lora) | Six way intent routing | 180 synthetic messages written for this run, 114 train, 24 dev, 42 test | Accuracy 1.000 | base gemma-2-2b-it 0.929, TF-IDF plus logistic regression 0.786, majority class 0.167 |

Caveats, all of which matter.

- The injection detector has a known false positive on a benign "please summarise this article" prompt. It is one layer, not a defense.
- The tweet fixture and the golden sample share sentence templates, so that test mostly measures how well the model learned the templates.
- The intent set is small and written by one author. A perfect score on 42 messages says little about real traffic.
- The 8B bet drafter is a config only. It was not trained and nothing is published for it.

## Reproducing the runs

The data prep scripts write jsonl splits to `data/splits`, which is gitignored. The configs expect them there. Needs the train extra and, for the injection set, network access to the Hugging Face dataset.

```
pip install -e '.[dev,train]'
python scripts/prep_injection.py
python scripts/prep_tweets.py
python scripts/intents_data.py
python -m foundry.encoders.train_injection foundry/encoders/configs/injection_deberta_v3_small.yaml
python -m foundry.finetune.sft foundry/finetune/configs/tweet_classifier_qwen25_15b_lora.yaml
python -m foundry.finetune.sft foundry/finetune/configs/intent_router_gemma2_2b_lora.yaml
```

The train extra expects transformers 5 or newer. On Apple silicon the trainer runs fp32 on MPS, and Gemma loads with eager attention. The evaluation scripts that produced the table are not in this repo yet. The test splits are fixed by the prep scripts, so the numbers can be recomputed from the published adapters.

## The release rule

The intent is that no rung ships without passing evalgate, and training metrics in a model card are not a gate. The three models above were not run through it, which is why they are described as experiments. llm-ops is meant to watch a rung live once one is deployed.

`foundry/hub/push.py` creates the Hub repo if it is missing and uploads the card, the train config and the adapter. It reads `HF_TOKEN` from the environment and refuses `.bin`, `.pt` and other pickle formats. Cards are rendered from templates in `foundry/hub/cards` and refuse to render with missing fields.

## Setup

The shared types come from sportsbook-contracts, which pip installs from git as a dependency.

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check .
pytest
```

Extras are `train`, `label` and `serve`. Tests never need them, never download a model and never touch the network. Copy `.env.example` for the keys that the labeling, serving and push scripts read.

## TODO

The injection detector config is exercised only by the run above. There is no committed fixture for it, so no test trains it end to end.

## License

MIT
