# Evaluation Overview

## Experiments Summary

| Exp | Name | Priority | Depends on |
|-----|------|----------|------------|
| Exp 1 | Content Quality (vs. Baselines) | **Critical** | generation |
| Exp 2 | Duplex Behavior (RIL, overlap) | **Critical** | generation |
| Exp 3 | Ablation (component + trigger) | High | generation |
| Exp 6 | Strategy & Think Analysis | Medium | generation |
| Exp 7 | User Study | Medium | generation |
| Exp 8 | Case Studies | Low | manual |

All automated experiments share the same **generation pipeline** — run inference first, then apply metrics.

---

## Generation Pipeline

### Input: Prepared Test Dataset

Pre-built by `Evalutation/prepare_eval_dataset.py` → `Evalutation/data/test_dataset.jsonl`

Each line is one (seeker turn, supporter turn) pair with fields:
- `conv_id`, `turn_id`, `source` (esconv / escot)
- `history`: list of previous turns (speaker + text)
- `seeker_wav`: path to current seeker audio
- `history_seeker_wavs`, `history_supporter_wavs`: prior audio for stream mode
- `gt_strategy`, `gt_think_trigger`, `gt_think`, `gt_response_text`

### Inference

Run `Evalutation/run_inference.py`:

```bash
# IMPORTANT: use CUDA_VISIBLE_DEVICES + --device cuda:0 for correct CUDA graph behavior
CUDA_VISIBLE_DEVICES=2 python Evalutation/run_inference.py \
  --history-mode text \
  --device cuda:0 \
  --out-dir Evalutation/results/personaplex_text \
  --num-convs 3   # sanity check; omit for full run
```

History modes:
- `text`  : GT history injected as system text prompt (PersonaPlex / our model)
- `audio` : previous seeker wavs prepended to input audio (model-agnostic)
- `both`  : text + audio
- `none`  : no history (context-free baseline)
- `stream`: full conversation as one continuous audio stream (Moshi-style)

**Audio layout for all modes**: `[history_or_seeker_audio | response_window (10 s silence)]`
The output wav covers only the response window (10 s).

### Timing reference

All frame indices in `turn_N_parsed.json` are **absolute** (from start of full input audio):

```
seeker_end_frame  = eval_offset_frames  (= first frame of response window)
response_start_frame = first non-PAD frame in response window (absolute)
RIL = (response_start_frame - seeker_end_frame) * 12.5 ms
```

### Output Storage

```
results/
  {model_name}/          # e.g. "personaplex_text", "personaplex_audio", "moshi_stream"
    run_config.json      # history_mode and other settings
    {conv_id}/
      turn_{N:02d}_output.wav          # RESPONSE_WINDOW_SEC of model audio
      turn_{N:02d}_output_tokens.json  # per-frame text tokens
      turn_{N:02d}_parsed.json         # timing + decoded text + GT fields
```

`turn_{N:02d}_parsed.json`:
```json
{
  "conv_id": "esconv_1011",
  "turn_id": 2,
  "history_mode": "text",
  "response_text": "I'm sorry to hear that...",
  "response_start_frame": 47,
  "eval_offset_frames": 42,
  "seeker_end_frame": 42,
  "gt_strategy": "Restatement or Paraphrasing",
  "gt_think_trigger": "sad",
  "gt_think": "The seeker is experiencing sadness...",
  "gt_response_text": "I'm sorry to hear that you are sad..."
}
```

---

## Metric Overview

| Metric | Exp | Tool | Input |
|--------|-----|------|-------|
| Consistency | 1 | GPT-4o judge | response_text + history |
| Informativeness | 1 | GPT-4o judge | response_text |
| Empathy | 1 | GPT-4o judge | response_text + seeker_text |
| Response Initiation Latency | 2, 3 | timing | response_start_frame, seeker_end_frame |
| Think Overlap Ratio | 2, 3 | timing | think_start/end_frame, seeker_end_frame |
| LLM strategy classification | 6 | GPT-4o classifier | response_text vs. gt_strategy annotation |
| Think–response consistency | 6 | NLI / cosine sim | think_text, response_text |
| Think diversity | 6 | distinct-n | think_text across conversations |

Detailed metric definitions: see per-experiment markdown files.
