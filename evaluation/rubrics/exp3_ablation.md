# Exp 3 — Ablation

## 3c — KV Cache Strategy Ablation (new)

Prove that the turn-level KV cache policy (discard audio, preserve text) is the optimal design.

### Four conditions

| Strategy | Audio KV | Text KV | Expected outcome |
|----------|----------|---------|-----------------|
| Full reset | Discard | Discard | Fast memory, but model has no history → quality ↓ |
| Full preserve | Keep | Keep | GPU memory grows O(T × N_turns) → OOM on long convs |
| **Ours** | Discard | Keep (re-inject) | Constant memory, full history → quality ✓ |
| No text history | Discard | Discard (no prompt) | Baseline: model ignores conversation history |

### Memory metric: GPU peak vs. turn count

Plot GPU memory (MB) as a function of conversation turn count N:
- **Full preserve**: memory grows linearly — each turn adds ~T_audio × hidden_dim per layer
- **Ours / Full reset**: flat — audio KV discarded, only text re-injected (constant token count per prompt)

```python
# Measurement: after each turn, record torch.cuda.max_memory_allocated()
import torch
torch.cuda.reset_peak_memory_stats()
# ... run turn ...
mem_mb = torch.cuda.max_memory_allocated() / 1e6
```

### Latency metric: prefill time vs. turn count

Text history prefill time as N grows:
- N×30 tokens → O(N) but parallelized → actual wall-clock growth is gentle (~50ms per 10 extra turns)
- Full audio re-encoding: O(N × audio_frames) → much steeper (sequential frame-by-frame encoding)

### Quality metric: consistency score vs. history retention

| Condition | Consistency (LLM judge) |
|-----------|------------------------|
| Ours (text history) | — |
| No text history | — (expected drop) |

**Key claim**: text-only history is sufficient; audio history adds memory cost without quality gain.
This justifies discarding audio KV between turns.

---

## 3a — Component Ablation

Isolate each component's contribution.

| Model variant | What is added/removed | Expected effect |
|---------------|----------------------|----------------|
| **Ours (PersonaPlex + CoT)** | — (base: no strategy token) | — |
| w/ strategy token | Add `<strategy>` before Think | Stronger conditioning; tests whether strategy label helps |
| w/o CoT (full ablation) | No Think stream at all | empathy↓, RIL↑ |

Same LLM judge pipeline as Exp 1. Strategy is an annotation only — this ablation tests whether it's worth adding it back.

---

## 3b — Think Trigger Ablation (critical)

Show that *when* Think starts matters.

| Condition | Think start | Expected RIL | Expected quality |
|-----------|-------------|-------------|-----------------|
| Turn-end trigger | After seeker finishes | High | OK (full think time) |
| Random trigger | Random word in seeker speech | Medium | Noisy |
| **Ours (predicted trigger)** | Model-predicted word | Low | Good |
| Oracle trigger | Ground-truth think_trigger label | Lowest | Best |

### Metrics

| Metric | Definition |
|--------|-----------|
| Think Completion Rate (TCR) | % of turns where Think finishes before seeker stops |
| Response Initiation Latency | Same as Exp 2 |
| Empathy score | LLM judge, same as Exp 1 |
| Consistency score | LLM judge, same as Exp 1 |

### Think Completion Rate

```python
# Think finishes before seeker stops = TOR > 0 and think_end_frame < seeker_end_frame
TCR = mean(think_end_frame < seeker_end_frame for each turn)
```

### Implementation Notes

- **Turn-end trigger**: modify inference to suppress Think until `seeker_end_frame` (set `think_start_frame = seeker_end_frame`)
- **Random trigger**: sample a random frame in `[0, seeker_end_frame]` as `think_start_frame`
- **Oracle trigger**: use `think_trigger` word position from dataset to set `think_start_frame`
- **Predicted trigger**: model's own prediction (standard inference)
