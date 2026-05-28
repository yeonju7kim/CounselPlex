"""
Post-process offline.py outputs into per-conversation evaluation artifacts.

For each conversation, given:
  - <input_dir>/<conv_id>.wav     : Stage-1 input WAV (seeker + silence)
  - <input_dir>/<conv_id>.json    : Stage-1 timeline metadata
  - <offline_dir>/<conv_id>.out.wav   : offline.py generated supporter audio
  - <offline_dir>/<conv_id>.out.json  : offline.py per-frame text tokens (12.5 fps)
  - source dataset metadata.json (auto-located from timeline.seeker_wavs)

Writes:
  - <output_dir>/<conv_id>.result.wav   : stereo, L=generated supporter, R=seeker input
  - <output_dir>/<conv_id>.result.txt   : readable per-turn transcript
  - <output_dir>/<conv_id>.result.json  : structured per-turn data
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import sphn

NON_TEXT_PIECES = {"PAD", "EPAD", "BOS", "EOS"}
THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"


def load_mono(path: Path) -> np.ndarray:
    pcm, sr = sphn.read(str(path))
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=0) if pcm.shape[0] > 1 else pcm[0]
    return pcm, sr


def _join(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def split_think_speech(seeker_pieces: list[str], silence_pieces: list[str]) -> tuple[str, str]:
    """Split a turn's pieces into (think, speech) by walking seeker+silence as one
    sequence. <think>/</think> flip state. Orphan </think> (close without prior
    open in this turn) is treated as a no-op so stray model-emitted close tags
    don't reclassify legitimate speech as think."""
    think_parts: list[str] = []
    speech_buf:  list[str] = []
    state = "speech"
    for p in list(seeker_pieces) + list(silence_pieces):
        if p in NON_TEXT_PIECES:
            continue
        if p == THINK_OPEN:
            state = "think"
            continue
        if p == THINK_CLOSE:
            state = "speech"
            continue
        (think_parts if state == "think" else speech_buf).append(p)
    return _join(think_parts), _join(speech_buf)


def split_think_with_forced(
    seeker_pieces: list[str], silence_pieces: list[str],
    seeker_forced: list[bool], silence_forced: list[bool],
) -> tuple[list[str], list[str], str]:
    """Like split_think_speech, but separates think blocks into (forced_thinks, free_thinks).
    Classifies a `<think>...</think>` block by the forced flag at the <think> open tag."""
    forced_thinks: list[str] = []
    free_thinks:   list[str] = []
    speech_buf:    list[str] = []
    state = "speech"
    cur_parts:  list[str] = []
    cur_forced = False
    pieces  = list(seeker_pieces) + list(silence_pieces)
    forced  = list(seeker_forced) + list(silence_forced)
    for p, f in zip(pieces, forced):
        if p in NON_TEXT_PIECES:
            continue
        if p == THINK_OPEN:
            state = "think"
            cur_parts = []
            cur_forced = f
            continue
        if p == THINK_CLOSE:
            if state == "think":
                (forced_thinks if cur_forced else free_thinks).append(_join(cur_parts))
                cur_parts = []
            state = "speech"
            continue
        if state == "think":
            cur_parts.append(p)
        else:
            speech_buf.append(p)
    if cur_parts:  # unclosed think
        (forced_thinks if cur_forced else free_thinks).append(_join(cur_parts))
    return forced_thinks, free_thinks, _join(speech_buf)


def load_metadata_text_map(seeker_wav_path: str) -> dict[str, str]:
    """Load source metadata.json and return {audio_filename: text}."""
    conv_dir = Path(seeker_wav_path).parent
    meta_path = conv_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return {t["audio"]: t.get("text", "") for t in meta.get("turns", [])}


def gather_text(wav_paths: list[str], text_map: dict[str, str]) -> str:
    parts = [text_map.get(Path(p).name, "") for p in wav_paths]
    return " ".join(p for p in parts if p).strip()


def build_one(
    conv_id: str,
    input_wav: Path,
    input_meta: Path,
    offline_wav: Path,
    offline_text: Path,
    output_dir: Path,
):
    with open(input_meta) as f:
        timeline = json.load(f)
    with open(offline_text) as f:
        pieces: list[str] = json.load(f)
    forced_path = offline_text.with_suffix("").with_name(offline_text.with_suffix("").name + ".forced.json")
    forced_mask: list[bool] | None = None
    if forced_path.exists():
        with open(forced_path) as f:
            forced_mask = json.load(f)
        if len(forced_mask) != len(pieces):
            print(f"[{conv_id}] WARN: forced_mask len={len(forced_mask)} != pieces len={len(pieces)} — ignoring")
            forced_mask = None

    in_pcm, in_sr = load_mono(input_wav)
    out_pcm, out_sr = load_mono(offline_wav)
    assert in_sr == out_sr, f"sample rate mismatch: input={in_sr}, output={out_sr}"
    sr = in_sr
    frame_rate = float(timeline["frame_rate"])
    samples_per_frame = sr / frame_rate

    n = max(in_pcm.shape[-1], out_pcm.shape[-1])
    if in_pcm.shape[-1] < n:
        in_pcm = np.concatenate([in_pcm, np.zeros(n - in_pcm.shape[-1], dtype=np.float32)])
    if out_pcm.shape[-1] < n:
        out_pcm = np.concatenate([out_pcm, np.zeros(n - out_pcm.shape[-1], dtype=np.float32)])
    stereo = np.stack([out_pcm[:n], in_pcm[:n]], axis=0)

    text_map: dict[str, str] = {}
    if timeline["turns"]:
        text_map = load_metadata_text_map(timeline["turns"][0]["seeker_wavs"][0])

    n_frames = len(pieces)

    def _frames(start_sample: int, end_sample: int) -> tuple[int, int]:
        fs = max(0, min(int(round(start_sample / samples_per_frame)), n_frames))
        fe = max(0, min(int(round(end_sample   / samples_per_frame)), n_frames))
        return fs, fe

    turn_results = []
    for turn in timeline["turns"]:
        sk_fs, sk_fe = _frames(turn["seeker_start_sample"],  turn["seeker_end_sample"])
        si_fs, si_fe = _frames(turn["silence_start_sample"], turn["silence_end_sample"])
        seeker_pieces  = pieces[sk_fs:sk_fe]
        silence_pieces = pieces[si_fs:si_fe]
        think_text, speech_text = split_think_speech(seeker_pieces, silence_pieces)
        forced_thinks: list[str] = []
        free_thinks:   list[str] = []
        if forced_mask is not None:
            seeker_forced  = forced_mask[sk_fs:sk_fe]
            silence_forced = forced_mask[si_fs:si_fe]
            forced_thinks, free_thinks, _ = split_think_with_forced(
                seeker_pieces, silence_pieces, seeker_forced, silence_forced
            )

        seeker_gt    = gather_text(turn["seeker_wavs"],    text_map)
        supporter_gt = gather_text(turn["supporter_wavs"], text_map)

        turn_results.append({
            "turn_id":              turn["turn_id"],
            "strategy":             turn.get("strategy"),
            "seeker_gt":            seeker_gt,
            "supporter_gt":         supporter_gt,
            "supporter_gen_think":  think_text,
            "supporter_gen_speech": speech_text,
            "supporter_gen_think_forced": forced_thinks,
            "supporter_gen_think_free":   free_thinks,
            "seeker_window_samples":  [turn["seeker_start_sample"],  turn["seeker_end_sample"]],
            "silence_window_samples": [turn["silence_start_sample"], turn["silence_end_sample"]],
            "seeker_window_frames":   [sk_fs, sk_fe],
            "silence_window_frames":  [si_fs, si_fe],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_out = output_dir / f"{conv_id}.result.wav"
    txt_out = output_dir / f"{conv_id}.result.txt"
    json_out = output_dir / f"{conv_id}.result.json"

    sphn.write_wav(str(wav_out), stereo, sr)

    with open(txt_out, "w") as f:
        f.write(f"=== conv_id: {conv_id} ===\n")
        mode_note = "  (inject-cum mode: forced=injected, model=self-emitted)" if forced_mask is not None else ""
        f.write(f"(L=generated supporter, R=seeker input; sr={sr}, frames={n_frames}){mode_note}\n\n")
        for r in turn_results:
            f.write(f"[Turn {r['turn_id']}]  (strategy: {r['strategy']})\n")
            f.write(f"  Seeker (GT)             : {r['seeker_gt']}\n")
            f.write(f"  Supporter (GT)          : {r['supporter_gt']}\n")
            if forced_mask is not None:
                inj  = " | ".join(r['supporter_gen_think_forced']) or "-"
                free = " | ".join(r['supporter_gen_think_free'])   or "-"
                f.write(f"  Supporter (think, INJ)  : {inj}\n")
                f.write(f"  Supporter (think, MODEL): {free}\n")
            else:
                f.write(f"  Supporter (think)       : {r['supporter_gen_think']}\n")
            f.write(f"  Supporter (speech)      : {r['supporter_gen_speech']}\n\n")

    with open(json_out, "w") as f:
        json.dump({
            "conv_id": conv_id,
            "sample_rate": sr,
            "frame_rate": frame_rate,
            "n_frames": n_frames,
            "turns": turn_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"[{conv_id}] {len(turn_results)} turns -> {wav_out.name}, {txt_out.name}, {json_out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="Stage-1 dir containing <conv>.wav and <conv>.json")
    ap.add_argument("--offline-dir", required=True,
                    help="Stage-2 dir containing <conv>.out.wav and <conv>.out.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conv-id", default=None,
                    help="If set, process only this single conversation")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    offline_dir = Path(args.offline_dir)
    output_dir = Path(args.output_dir)

    if args.conv_id is not None:
        conv_ids = [args.conv_id]
    else:
        conv_ids = sorted(p.stem for p in input_dir.glob("*.wav"))

    for cid in conv_ids:
        in_wav  = input_dir   / f"{cid}.wav"
        in_meta = input_dir   / f"{cid}.json"
        out_wav = offline_dir / f"{cid}.out.wav"
        out_txt = offline_dir / f"{cid}.out.json"
        missing = [p for p in (in_wav, in_meta, out_wav, out_txt) if not p.exists()]
        if missing:
            print(f"[{cid}] SKIP — missing: {[str(m) for m in missing]}")
            continue
        build_one(cid, in_wav, in_meta, out_wav, out_txt, output_dir)


if __name__ == "__main__":
    main()
