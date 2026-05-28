# Evaluation

Pipeline for reproducing the M5 content-quality scores reported in the paper.

## What this measures

For every supporter turn the LLM-as-judge (GPT-4.1) rates the response on five 1-5 dimensions:

| dim | what it captures |
|---|---|
| `empathy` | does the response acknowledge the seeker's emotion? |
| `coherence` | is it on-topic and well-formed? |
| `context_recalling` | does it reference earlier turns correctly? |
| `strategy_appropriateness` | is the ESConv strategy choice fitting? |
| `stressor_awareness` | does it engage with the seeker's underlying stressor? |

Two judges run independently:
- **`run_judge.py`** — scores **non-silent** turns on the 5 dims
- **`run_judge_silence.py`** — scores **silent** turns on a single `silence_appropriateness` axis (1-5)

`aggregate.py` then combines them into three views (silence excluded / silence = 0 / silence = judged).

## End-to-end pipeline

```text
inference outputs               build_eval_results.py             run_judge.py + run_judge_silence.py            aggregate.py
(moshi.offline .out.wav)  ──→   per-conv .result.{wav,txt,json}  ──→   per-conv .eval.json / .silence.json   ──→   final M5 table
```

### Step 0 — Set your OpenAI key

```bash
export OPENAI_API_KEY=sk-...
```

### Step 1 — Run inference on the test set

Use the test wavs in `data/test_dataset.jsonl` (535 conversations, 6,278 turns). Each conversation needs to be rendered into a single `<conv>.wav` and then fed through `moshi.offline` per the [main README](../README.md#offline-inference-batch).

### Step 2 — Build per-conversation result files

```bash
python evaluation/build_eval_results.py \
  --input-dir   path/to/conversation_wavs/ \
  --offline-dir path/to/moshi_offline_outputs/ \
  --output-dir  path/to/results/
```

Outputs `<conv_id>.result.{wav,txt,json}` per conversation.

### Step 3 — Run the two judges

```bash
# 5-dim judge on non-silent turns
python evaluation/run_judge.py \
  --results-dir path/to/results/ \
  --out-dir     path/to/eval/

# silence-appropriateness judge on silent turns
python evaluation/run_judge_silence.py \
  --results-dir path/to/results/ \
  --out-dir     path/to/eval_silence/
```

Both judges default to `gpt-4.1` (override with `--judge-model`). Add `--max-dialogues N` to limit.

### Step 4 — Aggregate

```bash
python evaluation/aggregate.py \
  --abs-dir     path/to/eval/ \
  --silence-dir path/to/eval_silence/ \
  --name        "CounselPlex"
```

Prints the three views (judged-only / silence-as-0 / silence-judged) plus silence-type breakdown.

## Rubrics

The exact prompts handed to the judge models live under `rubrics/`. `exp1_content_quality.md` is the canonical M5 rubric used in `run_judge.py`. The others (`exp2_duplex_behavior.md`, `exp3_ablation.md`, `exp6_strategy_think.md`) document additional analyses referenced in the paper appendix.

## Full-Duplex Bench

The paper also reports Full-Duplex-Bench metrics (turn-taking, pause handling, backchannel, etc.). Those use the upstream [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) evaluation pipeline — not duplicated here.
