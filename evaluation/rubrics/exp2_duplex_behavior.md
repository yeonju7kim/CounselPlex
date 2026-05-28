# Exp 2 — Duplex Behavior

## Goal

Show how the CoT Think + think_trigger mechanism affects real-time behavior.

## Metrics

### Response Initiation Latency (RIL)

Time from seeker's last word to supporter's first audio token.

```
RIL = (response_start_frame - seeker_end_frame) × frame_duration_ms
```

- `seeker_end_frame`: last non-PAD frame in the user audio channel
- `response_start_frame`: first non-PAD frame in the agent audio channel
- Frame duration: 12.5 ms (80 frames/sec, Moshi standard)

Expected pattern:
```
No CoT              : RIL ~ 500–800 ms
CoT, no trigger     : RIL ~ 1000+ ms  (think must happen after seeker stops)
CoT + trigger (ours): RIL ~ 100–300 ms (think overlaps seeker speech)
```

### Think Overlap Ratio (TOR)

Fraction of Think duration that overlaps with seeker speech (think happened while seeker was still speaking).

```
overlap_frames = min(think_end_frame, seeker_end_frame) - think_start_frame
TOR = overlap_frames / (think_end_frame - think_start_frame)
```

- TOR = 1.0: Think finished entirely while seeker was speaking → zero dead time
- TOR = 0.0: Think started only after seeker stopped (no overlap, maximum latency)

### How to Extract Timing from output.json

`output.json` has per-frame tokens aligned to audio frames.
- `think_start_frame`: frame index of first non-PAD token in think region
- `think_end_frame`: last frame of think region
- `response_start_frame`: first frame of response audio

The model's text channel emits `<think>` token when think starts.
→ `think_start_frame` = frame index of `<think>` token in output.json.

The seeker's end frame can be read from the input wav duration:
```python
seeker_end_frame = int(seeker_wav_duration_sec * 80)  # 80 frames/sec
```

## Key Comparison Table (to fill after experiments)

| Condition | RIL (ms) | TOR |
|-----------|----------|-----|
| No CoT | — | N/A |
| CoT, no trigger (turn-end) | — | 0.0 |
| CoT + predicted trigger (ours) | — | — |
| CoT + oracle trigger | — | 1.0 (upper bound) |

## Notes

- RIL measured on test set turns where seeker speaks ≥ 3 seconds (short utterances are less meaningful)
- Turns where model does not generate a response (edge case) are excluded
