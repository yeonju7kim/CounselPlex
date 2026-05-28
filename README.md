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
- `<id>.out.wav` — model's audio response (mono, 24 kHz)
- `<id>.out.json` — per-frame text tokens emitted by the model
- `<id>.out.forced.json` — per-frame mask (`true` = token was CUM-injected); written only when `--inject-cum` is on

## Online inference (WebSocket server)

```bash
python -m moshi.server_counselplex \
  --moshi-weight  checkpoints/counselplex.pt  \
  --mimi-weight   checkpoints/mimi.safetensors \
  --tokenizer     checkpoints/tokenizer.model  \
  --voice-prompt-dir checkpoints/voices       \
  --host 0.0.0.0  --port 8998
```

Connect a Moshi-compatible client (sends 24 kHz mono PCM frames, receives the same). The voice prompt and text prompt are negotiated per-session via the WebSocket handshake — the client tells the server which voice file (e.g. `woman_supporter.wav`) and what system prompt (e.g. `"You are a counselor."`) to use for that session.

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

## License

MIT (see `LICENSE`). Built on top of [Moshi](https://github.com/kyutai-labs/moshi) (Kyutai) and PersonaPlex (NVIDIA).
