from dataclasses import dataclass

TABLE_HEADER = (
    '| Rung | Tier | Task | Model | Owner repo | Latency | Cost |\n'
    '| --- | --- | --- | --- | --- | --- | --- |'
)


@dataclass(frozen=True)
class Rung:
    rung: int
    tier: str
    task: str
    model: str
    owner_repo: str
    latency_note: str
    cost_note: str
    replaces: str


LADDER: tuple[Rung, ...] = (
    Rung(
        rung=1,
        tier='Rules',
        task='Obvious feed filtering and market keyword routing',
        model='Regex and keyword tables',
        owner_repo='the consuming service',
        latency_note='Microseconds, in process',
        cost_note='Free',
        replaces='Paid calls that only checked whether a tweet mentions a tracked team',
    ),
    Rung(
        rung=2,
        tier='Classical ML',
        task='Tweet market impact baseline',
        model='TF-IDF plus engineered features, logistic regression or xgboost',
        owner_repo='model-foundry',
        latency_note='About a millisecond on CPU',
        cost_note='Free after training, trains in seconds',
        replaces='Frontier triage calls on clearly irrelevant or clearly routine tweets',
    ),
    Rung(
        rung=3,
        tier='Small encoder',
        task='Prompt injection detection on inbound text',
        model='DeBERTa v3 small with a classifier head',
        owner_repo='model-foundry',
        latency_note='Tens of milliseconds on one GPU or CPU',
        cost_note='Fixed instance cost, no per token fee',
        replaces='Guard prompts that asked a frontier model to spot injections',
    ),
    Rung(
        rung=4,
        tier='Fine tuned small LLM',
        task='Tweet classification, bet brief drafting, intent routing',
        model='Qwen2.5 1.5B, Llama 3.1 8B and Gemma 2 2B with LoRA adapters',
        owner_repo='model-foundry',
        latency_note='Low hundreds of milliseconds on vLLM, one host per base model',
        cost_note='Small GPUs per base, adapters keep per task cost low',
        replaces='Frontier classification, drafting and routing calls with a fixed output shape',
    ),
    Rung(
        rung=5,
        tier='Frontier',
        task='Open ended reasoning and the hard residual',
        model='Paid frontier models',
        owner_repo='the agent services',
        latency_note='Seconds',
        cost_note='Per token, highest',
        replaces='Nothing. This is where work lands when the cheaper rungs abstain',
    ),
)


def render_table(ladder: tuple[Rung, ...] = LADDER) -> str:
    rows = [
        f'| {r.rung} | {r.tier} | {r.task} | {r.model} | {r.owner_repo} '
        f'| {r.latency_note} | {r.cost_note} |'
        for r in ladder
    ]
    return '\n'.join([TABLE_HEADER, *rows])
