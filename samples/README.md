# Sample inference outputs

Three CounselPlex conversations selected jointly on two criteria so that what you hear sounds good end-to-end, not just at one cherry-picked turn:

1. **Response rate ≥ 93 %** — CounselPlex actually speaks at almost every supporter turn (no long silent stretches).
2. **High mean GPT-4.1 score across ≥ 4 independent judges** — quality is consistent, not a single lucky rating.

| Sample | Turns | Response rate | Mean GPT-4.1 score | # judges | Peak score |
|---|---|---|---|---|---|
| [`escot_test_1549/`](escot_test_1549/) | 9 | 100 % | 18.0 / 20 | 4 | 20 |
| [`escot_test_1554/`](escot_test_1554/) | 14 | 93 % | 17.75 / 20 | 4 | 18 |
| [`escot_test_1257/`](escot_test_1257/) | 15 | 100 % | 17.0 / 20 | 6 | 19 |

## Per-sample files

```
{sample}/
├── input.wav     # seeker audio: full multi-turn conversation
├── input.json    # conversation metadata (per-turn timestamps, GT supporter text)
├── output.wav    # full-duplex stereo: ch 0 = seeker (input), ch 1 = CounselPlex
├── output.json   # per-frame text tokens emitted by the model
└── output.txt    # human-readable transcript (think + speech per turn)
```

The model was run with the paper's settings:
`--temp-text 0.3 --repetition-penalty 1.3 --repetition-window 64 --repetition-skip-forced --inject-cum --seed 42`

To reproduce a sample, pass `input.wav` to `moshi.offline` per the [main README](../README.md#offline-inference-batch).
