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

The large weight files are **not** included in this repo (GitHub file-size limits). Two downloads are needed:

**1. CounselPlex finetuned weights (16 GB)** — from [our HF Hub repo](https://huggingface.co/yeonju7kim/CounselPlex):
```bash
hf download yeonju7kim/CounselPlex counselplex.pt \
  --local-dir checkpoints/
```

**2. Mimi audio codec (367 MB)** — from [Kyutai's Moshi release](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (this is the upstream component, not finetuned by us):
```bash
hf download kyutai/moshiko-pytorch-bf16 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  --local-dir checkpoints/
mv checkpoints/tokenizer-e351c8d8-checkpoint125.safetensors checkpoints/mimi.safetensors
```

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
- `<id>.out.wav` — model's audio response (stereo: ch0 = user input, ch1 = model)
- `<id>.out.json` — per-frame text tokens emitted by the model

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
