# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline inference entrypoint for PersonaPlex that mirrors server.py behavior without a WebSocket server.

High-level flow:
- Load Mimi encoders/decoders, Moshi LM, and tokenizer (same as server.py)
- Warmup to initialize CUDA graphs and streaming state
- Prompt phase: load system text tokens and a voice prompt WAV (agent side)
- Streaming-like phase: feed user audio frames from a WAV file into the "input" channels,
  autoregressively sample text + agent audio channels each step, and decode audio frames
- Concatenate generated frames and write an output WAV matching the input duration

This script reuses helpers from lm.py (load_audio, _iterate_audio, encode_from_sphn) to
keep parity with voice-prompt feeding logic in the server.
"""

import argparse
import os
import tarfile
from pathlib import Path
import json
from typing import Optional, List

import numpy as np
import torch
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders, LMGen, MimiModel
from .models.lm import load_audio as lm_load_audio
from .models.lm import _iterate_audio as lm_iterate_audio
from .models.lm import encode_from_sphn as lm_encode_from_sphn


def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed: int):
    """Seed torch, CUDA, numpy, and Python RNG for reproducible runs.

    Matches the seeding strategy in server.py.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: str, frame_size: int):
    """Run a short warmup loop to initialize CUDA graphs and streaming state.

    Replicates the same warmup behavior as server.py: zeros → encode → LMGen.step → decode.
    """
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            # Decode agent audio channels to ensure decode graphs/states are primed
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_tokens_to_pcm(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, tokens: torch.Tensor) -> np.ndarray:
    """Decode a single step of model tokens to PCM using Mimi.

    tokens is shaped [B, dep_q+1, 1]; channels 1..dep_q are the agent audio codebooks.
    Returns a 1D float32 numpy array (mono) for the current frame.
    """
    pcm = mimi.decode(tokens[:, 1:9])
    _ = other_mimi.decode(tokens[:, 1:9])
    pcm = pcm.detach().cpu().numpy()[0, 0]
    return pcm


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    log("info", "retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        log("info", f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def setup_state(
    text_prompt: str,
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
    max_context: Optional[int] = None,
    pad_logit_bias: float = 0.0,
    repetition_penalty: float = 1.0,
    repetition_window: int = 32,
    repetition_skip_forced: bool = False,
) -> dict:
    """Load mimi/tokenizer/LM, build LMGen, warm up, set voice + text prompts.

    Returns a state dict reused across multiple input WAVs by process_one().
    """
    hf_hub_download(hf_repo, "config.json")

    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload, context=max_context)
    lm.eval()
    log("info", "moshi loaded")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
        pad_logit_bias=pad_logit_bias,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        repetition_skip_forced=repetition_skip_forced,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    if voice_prompt_path.endswith('.pt'):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )

    return {
        "mimi": mimi,
        "other_mimi": other_mimi,
        "lm_gen": lm_gen,
        "text_tokenizer": text_tokenizer,
    }


def _compute_vad_mask(audio_np, sample_rate: int, frame_rate: float):
    """Per-Mimi-frame VAD aligned exactly to Mimi's 80ms grid (no drift).

    Each Mimi frame = 1920 samples @ 24kHz = 80ms = 1280 samples @ 16kHz.
    Silero v5 requires exactly 512 samples @ 16kHz (32ms); we slide it
    inside each 80ms window (3 sub-chunks: 0/256/512..) and OR the results.
    Falls back to RMS energy with EXACT 1920-sample frames if Silero fails.
    """
    import torch
    samples_per_mimi_24k = int(round(sample_rate / frame_rate))        # 1920 @ 24k/12.5
    n_frames_total = int(audio_np.shape[-1] // samples_per_mimi_24k) + 1
    try:
        vad_model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad',
            force_reload=False, trust_repo=True, verbose=False
        )
        vad_model.reset_states()

        import torchaudio
        audio_t = torch.from_numpy(audio_np).float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)
        audio_16k = torchaudio.functional.resample(audio_t, sample_rate, 16000)[0]   # [T]
        samples_per_mimi_16k = 1280                                    # 80ms @ 16k
        silero_chunk_16k     = 512                                     # Silero v5 requirement
        # how many 512-sample sub-chunks per 80ms window
        n_sub = max(1, samples_per_mimi_16k // silero_chunk_16k)       # 2
        is_active = np.zeros(n_frames_total, dtype=bool)
        for i in range(n_frames_total):
            start = i * samples_per_mimi_16k
            active = False
            for k in range(n_sub):
                s = start + k * silero_chunk_16k
                chunk = audio_16k[s : s + silero_chunk_16k]
                if chunk.shape[-1] < silero_chunk_16k:
                    chunk = torch.nn.functional.pad(chunk, (0, silero_chunk_16k - chunk.shape[-1]))
                with torch.no_grad():
                    prob = vad_model(chunk, 16000).item()
                if prob > 0.5:
                    active = True
                    break
            is_active[i] = active
        return is_active
    except Exception as e:
        log("warning", f"Silero VAD failed ({type(e).__name__}: {e}); falling back to RMS")
        is_active = np.zeros(n_frames_total, dtype=bool)
        for i in range(n_frames_total):
            chunk = audio_np[..., i * samples_per_mimi_24k : (i + 1) * samples_per_mimi_24k]
            if chunk.size == 0:
                break
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_active[i] = rms > 0.005
        return is_active


def process_one(
    state: dict,
    input_wav: str,
    output_wav: str,
    output_text: str,
    seed: Optional[int] = None,
    inject_cum: bool = False,
    cum_delay_frames: int = 3,   # frames after VAD active → inject. With ~3-frame VAD lag, total ≈ 6f = 0.5s after speech starts (matches training cum_delay=6).
):
    """Run inference for a single input WAV using preloaded state.

    Resets streaming state and replays the system prompts so each call is
    independent of prior calls within the same process.
    """
    if seed is not None and seed != -1:
        seed_all(seed)

    mimi = state["mimi"]
    other_mimi = state["other_mimi"]
    lm_gen = state["lm_gen"]
    text_tokenizer = state["text_tokenizer"]

    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    # Reset repetition-penalty history (per-conv)
    lm_gen._text_history = []
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    sample_rate = mimi.sample_rate
    user_audio = lm_load_audio(input_wav, sample_rate)

    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []
    generated_forced_mask: List[bool] = []  # per token: True if force-injected (CUM)
    total_target_samples = user_audio.shape[-1]

    # ── Force-inject CUM state (self-detecting via VAD) ───────────────────
    TS_ID, TE_ID = 32000, 32001
    CUM_DELAY_FRAMES = cum_delay_frames   # frames after VAD-active to inject (with ~3-frame VAD lag → ~1s after speech for default 9)
    is_active = _compute_vad_mask(user_audio, sample_rate, 12.5) if inject_cum else None
    if inject_cum:
        log("info", f"inject_cum=ON, vad active frames: {int(is_active.sum())}/{len(is_active)}")
    cum_current_phrase: str = ""        # v7: single master phrase, replaced on each capture
    inj_state = "idle"                   # idle / injecting_cum / collecting_new
    inj_buf: List[int] = []
    inj_pos = 0
    new_buf: List[int] = []
    seeker_in_turn = False               # True between long-silence boundaries
    silence_run = 0                       # consecutive silent frames
    SILENCE_BREAK_FRAMES = 12            # >=1s silence → new seeker turn boundary
    seeker_active_count = 0
    seeker_turn_idx = -1                 # increments each new seeker turn (False→True)
    last_injected_turn = -1              # turn idx where we last injected (prevents double-inject in same turn)
    global_step = 0
    import torch as _torch
    _device = lm_gen.lm_model.text_emb.weight.device

    def _build_cum_inject_buf() -> List[int]:
        # v7/v8: cum is a single refined phrase. Training has NO "topic:" prefix,
        # so inject must match exactly (just the phrase between TS/TE).
        body = text_tokenizer.encode(cum_current_phrase)  # type: ignore
        return [TS_ID] + body + [TE_ID]

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
        max_batch=1,
    ):
        steps = user_encoded.shape[-1]
        for c in range(steps):
            step_in = user_encoded[:, :, c : c + 1]

            # ── 1) Detect user-active transitions (with silence smoothing) ──
            if inject_cum:
                user_active = bool(is_active[global_step]) if global_step < len(is_active) else False
                if user_active:
                    silence_run = 0
                    if not seeker_in_turn:
                        seeker_in_turn = True
                        seeker_active_count = 1
                        seeker_turn_idx += 1   # new seeker turn started
                    else:
                        seeker_active_count += 1
                else:
                    silence_run += 1
                    if silence_run >= SILENCE_BREAK_FRAMES and seeker_in_turn:
                        seeker_in_turn = False
                        seeker_active_count = 0

                # Inject CUM N frames after seeker turn starts — once per turn,
                # whenever there's something to remind (regardless of whether
                # a new topic was just captured).
                if (inj_state == "idle"
                        and seeker_in_turn
                        and seeker_active_count == CUM_DELAY_FRAMES
                        and cum_current_phrase
                        and seeker_turn_idx > last_injected_turn):
                    inj_buf = _build_cum_inject_buf()
                    inj_pos = 0
                    inj_state = "injecting_cum"
                    last_injected_turn = seeker_turn_idx
                    log("info", f"[inject] cum at step {global_step} (turn {seeker_turn_idx}): {cum_current_phrase!r}")

            # ── 2) Step (with optional force) ────────────────────────────
            this_step_forced = False
            if inject_cum and inj_state == "injecting_cum":
                this_step_forced = True
                forced_id = inj_buf[inj_pos]
                forced = _torch.tensor([forced_id], device=_device)
                tokens = lm_gen.step(step_in, text_token=forced)
                inj_pos += 1
                if inj_pos >= len(inj_buf):
                    inj_state = "idle"
            else:
                tokens = lm_gen.step(step_in)

            if tokens is None:
                global_step += 1
                continue
            pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
            generated_frames.append(pcm)
            text_token = tokens[0, 0, 0].item()

            # ── 3) Capture model-emitted new topic in gap region ─────────
            if inject_cum:
                _sp_vocab = text_tokenizer.vocab_size()  # type: ignore
                if inj_state == "idle" and (not user_active) and text_token == TS_ID:
                    inj_state = "collecting_new"
                    new_buf = []
                elif inj_state == "collecting_new":
                    if text_token == TE_ID:
                        inj_state = "idle"
                        # Filter to in-vocab tokens only (model may emit stray TS/special)
                        safe_buf = [t for t in new_buf if 0 < t < _sp_vocab]
                        if safe_buf:
                            try:
                                decoded = text_tokenizer.decode(safe_buf)  # type: ignore
                            except Exception as e:
                                log("warning", f"[capture] decode failed ({type(e).__name__}); skipping")
                                continue
                            decoded = decoded.replace("▁", " ").strip()
                            for pref in ("new:", "topic:", "new", "topic"):
                                if decoded.startswith(pref):
                                    decoded = decoded[len(pref):].lstrip(": ").strip()
                                    break
                            if decoded and decoded != cum_current_phrase:
                                cum_current_phrase = decoded
                                log("info", f"[capture] master phrase: {decoded!r}")
                            elif decoded:
                                log("info", f"[capture] dup skipped: {decoded!r}")
                    elif 0 < text_token < _sp_vocab:
                        new_buf.append(text_token)
                    # else: stray out-of-vocab token (TS, TE handled above), ignore

            # ── 4) Log/append text token ─────────────────────────────────
            # Was this token force-injected? (inj_state was injecting_cum BEFORE inj_pos
            # advanced at step time; track via "this step was inside an inject buffer")
            was_forced = inject_cum and inj_state == "injecting_cum"  # we just consumed forced token; state may have flipped to idle
            # The above is set BEFORE step() flips state, but we already mutated inj_state.
            # We capture via a small helper: any token written during the injecting_cum
            # window is forced. We track via 'just_forced' set right above when forcing.
            sp_vocab = text_tokenizer.vocab_size()  # type: ignore
            if text_token in (0, 3):
                text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                log("info", f"text token '{text_token_map[text_token]}'")
                generated_text_tokens.append(text_token_map[text_token])
            elif text_token >= sp_vocab:
                extra_names = {32000: '<think>', 32001: '</think>'}
                name = extra_names.get(text_token, f'<extra_{text_token}>')
                log("info", f"text token '{name}'")
                generated_text_tokens.append(name)
            else:
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                log("info", f"text token '{_text}'")
                generated_text_tokens.append(_text)
            generated_forced_mask.append(this_step_forced)
            global_step += 1

    if len(generated_frames) == 0:
        log("error", "No audio frames were generated. Check input file and configuration.")
        return

    output_pcm = np.concatenate(generated_frames, axis=-1)
    if output_pcm.shape[-1] > total_target_samples:
        output_pcm = output_pcm[:total_target_samples]
    elif output_pcm.shape[-1] < total_target_samples:
        pad_len = total_target_samples - output_pcm.shape[-1]
        output_pcm = np.concatenate(
            [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
        )

    sphn.write_wav(output_wav, output_pcm, sample_rate)
    log("info", f"Wrote output audio to {output_wav}")

    with open(output_text, "w") as file:
        json.dump(generated_text_tokens, file, ensure_ascii=False)
    # Also save the per-token forced_mask alongside (True = force-injected CUM)
    if inject_cum:
        forced_path = Path(output_text).with_suffix("")
        forced_path = forced_path.with_name(forced_path.name + ".forced.json")
        with open(forced_path, "w") as f2:
            json.dump(generated_forced_mask, f2)
    log("info", f"Wrote output text to {output_text}")


def main():
    """Parse CLI args and run offline inference."""
    parser = argparse.ArgumentParser(
        description="Offline inference from WAV input using Moshi server components."
    )
    parser.add_argument(
        "--input-wav", type=str, help="Single-input mode: path to input WAV file (user audio)"
    )
    parser.add_argument(
        "--output-wav", type=str, help="Single-input mode: path to output WAV to write"
    )
    parser.add_argument(
        "--output-text", type=str, help="Single-input mode: path to output JSON of agent text tokens"
    )
    parser.add_argument(
        "--input-dir", type=str,
        help="Batch mode: process every *.wav under this directory using a single model load."
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Batch mode: write <stem>.out.wav and <stem>.out.json into this directory."
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Batch mode: skip a conversation if both output files already exist (default: on)."
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Batch mode: regenerate even if outputs exist."
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Batch mode: iterate input wavs in reverse order (useful for 2-process splits)."
    )
    parser.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.", type=str, help="Text prompt")

    parser.add_argument(
        "--voice-prompt", required=True, type=str, help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'woman_supporter.wav')."
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from -voice-prompt arg will be joined with this directory path."
        )
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to pre-trained model repo)",
    )

    # Runtime / sampling controls (mirror UI semantics)
    parser.add_argument(
        "--temp-audio", type=float, default=0.8, help="Audio sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7, help="Text sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250, help="Audio top-k sampling (default: 250)"
    )
    parser.add_argument(
        "--topk-text", type=int, default=25, help="Text top-k sampling (default: 25)"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling (greedy decoding)"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Text repetition penalty (1.0=off, 1.2-1.3 mild, >=1.5 strong). "
             "Penalizes recently-emitted non-special text tokens."
    )
    parser.add_argument(
        "--repetition-window", type=int, default=32,
        help="How many recent text tokens to track for repetition penalty (default 32)."
    )
    parser.add_argument(
        "--repetition-skip-forced", action="store_true",
        help="When set, exclude force-injected text tokens (cum overlay) from the "
             "repetition history. Recommended with --inject-cum to avoid penalizing "
             "the model's own vocabulary just because the cum reminder used those words."
    )
    parser.add_argument(
        "--inject-cum", action="store_true",
        help="v6: auto-detect seeker turn starts (VAD on input wav) and force-inject "
             "the model's previously-emitted <think> topics as a cumulative reminder "
             "1s after each seeker turn begins. Model still freely generates new "
             "topics in gap regions; those get captured into the running cum list."
    )
    parser.add_argument(
        "--cum-delay-frames", type=int, default=3,
        help="Frames after VAD detects seeker-active before injecting cum overlay. "
             "With ~3-frame VAD lag, 3 ≈ 0.5s after speech actually starts (matches training cum_delay=6)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'."
    )
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables)")
    parser.add_argument(
        "--max-context", type=int, default=None,
        help="Override attention context window (frames). Sets both attention mask "
             "range and RingKVCache capacity. Use 2048 to match finetune training cap."
    )
    parser.add_argument(
        "--pad-bias", type=float, default=0.0,
        help="Subtract this value from PAD token logit before text sampling. "
             "Positive value reduces PAD likelihood (helps escape silence attractor under greedy)."
    )

    args = parser.parse_args()

    single = bool(args.input_wav or args.output_wav or args.output_text)
    batch  = bool(args.input_dir or args.output_dir)
    if single and batch:
        parser.error("specify either single-input (--input-wav/--output-wav/--output-text) "
                     "or batch (--input-dir/--output-dir), not both")
    if single and not (args.input_wav and args.output_wav and args.output_text):
        parser.error("single-input mode requires --input-wav, --output-wav, and --output-text")
    if batch and not (args.input_dir and args.output_dir):
        parser.error("batch mode requires both --input-dir and --output-dir")
    if not single and not batch:
        parser.error("must specify either single-input or batch mode")

    # If --voice-prompt-dir is omitted, voices.tgz is downloaded from HF and extracted.
    voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    log("info", f"voice_prompt_dir = {voice_prompt_dir}")

    # Join basename with directory (DO NOT mutate args.voice_prompt)
    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    # Normalize greedy flag behavior (True if present, False otherwise)
    greedy = bool(args.greedy)

    if batch:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for wav in sorted(input_dir.glob("*.wav"), reverse=args.reverse):
            out_wav = output_dir / f"{wav.stem}.out.wav"
            out_txt = output_dir / f"{wav.stem}.out.json"
            if args.skip_existing and out_wav.exists() and out_txt.exists():
                log("info", f"[{wav.stem}] cached, skipping")
                continue
            jobs.append((wav.stem, str(wav), str(out_wav), str(out_txt)))
        if not jobs:
            log("info", "nothing to do (all outputs cached or input dir empty)")
            return
    else:
        jobs = [(Path(args.input_wav).stem, args.input_wav, args.output_wav, args.output_text)]

    with torch.no_grad():
        state = setup_state(
            text_prompt=args.text_prompt,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
            max_context=args.max_context,
            pad_logit_bias=args.pad_bias,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            repetition_skip_forced=args.repetition_skip_forced,
        )
        n_ok = 0
        n_fail = 0
        for cid, in_wav, out_wav, out_txt in jobs:
            log("info", f"[{cid}] generating ({n_ok + n_fail + 1}/{len(jobs)})")
            try:
                process_one(state, in_wav, out_wav, out_txt, seed=args.seed,
                            inject_cum=args.inject_cum,
                            cum_delay_frames=args.cum_delay_frames)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                log("error", f"[{cid}] failed: {e!r}")
        log("info", f"done: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()