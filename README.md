# CounselPlex

CounselPlex is a CoT-augmented full-duplex spoken counseling model, built on PersonaPlex (Moshi-based 7B). It performs real-time emotional support dialogue with internal `<think>` reasoning between turns.

## Repository contents

```
CounselPlex/
├── moshi/                       # inference package (PyTorch)
│   ├── offline.py               # batch / file-based inference
│   ├── server_counselplex.py    # WebSocket streaming server (online)
│   ├── models/, modules/, ...   # model implementation
├── checkpoints/
│   ├── counselplex.pt           # 8.4B model weights
│   ├── mimi.safetensors         # audio codec
│   ├── tokenizer.model          # SentencePiece text tokenizer
│   └── voices/                  # voice prompt embeddings + sample wav
├── evaluation/                  # reproduce the paper's M5 scores
│   ├── run_judge.py             # GPT-4.1 5-dim judge (non-silent turns)
│   ├── run_judge_silence.py     # silence-appropriateness judge
│   ├── aggregate.py             # final M5 table
│   ├── build_eval_results.py    # moshi.offline output → per-conv result.json
│   ├── rubrics/                 # exact prompts handed to the judge
│   └── data/test_dataset.jsonl  # 535 conv / 6,278 turn test split
├── samples/                     # 3 end-to-end inference examples (input + output wav)
├── pyproject.toml / requirements.txt
└── LICENSE
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.10 and a CUDA-capable GPU (≥ 24 GB recommended).

## Download model weights

The large weight files are **not** included in this GitHub repo (file-size limits). Download both from [our HuggingFace Hub repo](https://huggingface.co/yeonju7kim/CounselPlex):

```bash
hf download yeonju7kim/CounselPlex \
  counselplex.pt mimi.safetensors \
  --local-dir checkpoints/
```

- `counselplex.pt` (16 GB) — the finetuned CounselPlex model (8.4 B parameters)
- `mimi.safetensors` (367 MB) — Mimi audio codec; the file is a verbatim copy of `tokenizer-e351c8d8-checkpoint125.safetensors` from Kyutai's [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (mirrored here for reproducibility — we did not retrain it)

Smaller assets (text tokenizer, voice prompts) are committed directly in this repo.

## Offline inference (batch)

```bash
python -m moshi.offline \
  --input-dir   path/to/wavs/                 \
  --output-dir  path/to/output/               \
  --moshi-weight  checkpoints/counselplex.pt  \
  --mimi-weight   checkpoints/mimi.safetensors \
  --tokenizer     checkpoints/tokenizer.model  \
  --voice-prompt-dir checkpoints/voices       \
  --voice-prompt  woman_supporter.wav         \
  --text-prompt   "You are a counselor."      \
  --temp-text     0.3                         \
  --repetition-penalty  1.3                   \
  --repetition-window   64                    \
  --repetition-skip-forced                    \
  --inject-cum                                \
  --seed 42
```

Each `<id>.wav` in the input directory is processed independently. Outputs:
- `<id>.out.wav` — full-duplex stereo at 24 kHz (ch 0 = seeker input, ch 1 = CounselPlex)
- `<id>.out.json` — per-frame text tokens emitted by the model
- `<id>.out.forced.json` — per-frame mask (`true` = token was CUM-injected); written only when `--inject-cum` is on

## Online inference (WebSocket server)

```bash
python -m moshi.server_counselplex \
  --moshi-weight  checkpoints/counselplex.pt  \
  --mimi-weight   checkpoints/mimi.safetensors \
  --tokenizer     checkpoints/tokenizer.model  \
  --voice-prompt-dir checkpoints/voices       \
  --voice-prompt  woman_supporter.wav         \
  --text-prompt   "You are a counselor."      \
  --inject-cum                                \
  --cum-delay-frames 3                        \
  --host 127.0.0.1 --port 8998
```

Open `http://127.0.0.1:8998` for the WebUI, or connect a Moshi-compatible client. Frames are 24 kHz mono PCM in both directions.

The voice and text prompts are fixed server-side via `--voice-prompt` / `--text-prompt` (the WebUI's selection screen is hidden — the same CounselPlex voice and counselor system prompt are used for every session).

`--inject-cum` turns on cross-turn stressor carrying in streaming mode (same mechanism as `moshi.offline`): the server runs Silero VAD per Mimi frame, captures the model's `<think>…</think>` master phrase whenever the seeker is silent, and force-injects that phrase at the start of every new seeker turn. When it fires you'll see lines like

```
[capture] master phrase: 'toxic work environment'
[inject]  cum at step 307 (turn 1): 'toxic work environment'
```

in the server terminal. Omit `--inject-cum` to disable it. `--cum-delay-frames` (default 3 ≈ 240 ms) controls how many VAD-active frames into a new seeker turn the injection fires.

## Hyperparameters used in the paper

| flag | value |
|---|---|
| `--temp-text` | 0.3 |
| `--repetition-penalty` | 1.3 |
| `--repetition-window` | 64 |
| `--repetition-skip-forced` | (on) |
| `--inject-cum` | (on) |
| `--seed` | 42 |

`--inject-cum` enables cross-turn stressor carrying via cumulative think-token injection; it is the inference-time switch matching the "inject-cum" training pattern.

## Reproducing the paper's evaluation

The full pipeline (inference → judge → aggregate) is documented in [`evaluation/README.md`](evaluation/README.md). High level:

```bash
export OPENAI_API_KEY=sk-...

# 1) run moshi.offline on the 535-conv test set (see evaluation/data/test_dataset.jsonl)
# 2) post-process raw outputs:
python evaluation/build_eval_results.py --input-dir … --offline-dir … --output-dir results/

# 3) judge:
python evaluation/run_judge.py         --results-dir results/ --out-dir eval/
python evaluation/run_judge_silence.py --results-dir results/ --out-dir eval_silence/

# 4) aggregate to the final M5 table:
python evaluation/aggregate.py --abs-dir eval/ --silence-dir eval_silence/ --name CounselPlex
```

Full-Duplex-Bench metrics use the upstream [FDB pipeline](https://github.com/DanielLin94144/Full-Duplex-Bench) — not duplicated here.

## License

MIT (see `LICENSE`). Built on top of [Moshi](https://github.com/kyutai-labs/moshi) (Kyutai) and PersonaPlex (NVIDIA).
