"""
Aggregate absolute eval (1-5 per dimension) with silence-appropriateness scores.

For each turn:
  - if the model responded:  use the existing 1-5 score on each of 4 dimensions.
  - if the model was silent: use the silence_appropriateness score (1-5), broadcast
                             to all 4 dimensions. Rationale: silence is a unified
                             counseling act (you can't separately rate the empathy
                             vs. coherence of an absence); a single appropriateness
                             score is the cleanest scalar to plug into each column.

Three views are reported per model:
  1. judged-only      : silent turns excluded (= original CSV columns)
  2. silence-as-0     : silent turns scored 0 on all dims (lower bound)
  3. silence-judged   : silent turns scored using silence_appropriateness (1-5)

Plus side breakdown: # silent turns by silence_type (avoidance / neutral / therapeutic).

Usage:
  python Evalutation/aggregate_absolute_with_silence.py
"""

import argparse
import json
from pathlib import Path
from collections import Counter

DIMS = ["empathy", "coherence", "context_recalling", "strategy_appropriateness", "stressor_awareness"]


def load_abs(abs_dir):
    turns = []
    for f in sorted(Path(abs_dir).glob("*.eval.json")):
        d = json.load(open(f))
        for t in d["turns"]:
            turns.append((d["conv_id"], t))
    return turns


def has_dim(turns, dim):
    return any(dim in t for _, t in turns if t.get("engagement") == 1)


def load_silence(silence_dir):
    """conv_id, turn_id -> {silence_appropriateness, silence_type, reason}"""
    sil = {}
    p = Path(silence_dir)
    if not p.exists():
        return sil
    for f in sorted(p.glob("*.silence.json")):
        d = json.load(open(f))
        conv = d["conv_id"]
        for t in d["silence_turns"]:
            sil[(conv, t["turn_id"])] = t
    return sil


def compute_means(abs_turns, sil_map):
    """Returns three dicts: judged_only, silence_as_0, silence_judged.
    Each maps dim -> mean across n_total."""
    n_total = len(abs_turns)
    judged = [(c, t) for c, t in abs_turns if t.get("engagement") == 1 and "empathy" in t]
    silent = [(c, t) for c, t in abs_turns if t.get("engagement") == 0]

    means_jo, means_s0, means_sj = {}, {}, {}
    for dim in DIMS:
        judged_vals = [t[dim] for _, t in judged if dim in t]
        if not judged_vals:
            means_jo[dim] = means_s0[dim] = means_sj[dim] = None
            continue
        sj_silent_vals = []
        for c, t in silent:
            entry = sil_map.get((c, t["turn_id"]))
            score = entry.get("silence_appropriateness") if entry else None
            if score is None:
                sj_silent_vals.append(0)   # fallback if silence wasn't judged
            else:
                sj_silent_vals.append(score)

        # Use the count of judged_vals (turns that actually have the dim) as denominator
        # for judged-only, but n_total (full sample) for silence-aware views.
        n_j = len(judged_vals)
        means_jo[dim] = sum(judged_vals) / n_j if n_j else 0
        # Pad missing-dim turns are not counted (judged_vals is already filtered).
        means_s0[dim] = sum(judged_vals) / n_total if n_total else 0
        means_sj[dim] = (sum(judged_vals) + sum(sj_silent_vals)) / n_total if n_total else 0

    # silence type breakdown
    sil_types = Counter()
    sil_scores = []
    for c, t in silent:
        entry = sil_map.get((c, t["turn_id"]))
        if entry:
            sil_types[entry.get("silence_type", "unknown")] += 1
            if "silence_appropriateness" in entry:
                sil_scores.append(entry["silence_appropriateness"])
        else:
            sil_types["not_judged"] += 1
    mean_sil = sum(sil_scores) / len(sil_scores) if sil_scores else None

    return {
        "n_total": n_total, "n_judged": len(judged), "n_silent": len(silent),
        "judged_only": means_jo, "silence_as_0": means_s0, "silence_judged": means_sj,
        "silence_types": dict(sil_types), "mean_silence_appr": mean_sil,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--abs-dir", required=True,
                   help="directory of *.eval.json from run_judge.py")
    p.add_argument("--silence-dir", required=True,
                   help="directory of *.silence.json from run_judge_silence.py")
    p.add_argument("--name", default="model",
                   help="display name for the row")
    args = p.parse_args()

    if not Path(args.abs_dir).exists():
        raise SystemExit(f"abs-dir not found: {args.abs_dir}")

    abs_turns = load_abs(args.abs_dir)
    sil_map = load_silence(args.silence_dir)
    r = compute_means(abs_turns, sil_map)
    r["name"] = args.name
    rows = [r]

    print(f"\n{'model':30s} {'n_t':>4s} {'STRR':>6s} {'sil_n':>5s} {'sil_appr':>9s}   "
          f"{'silence_type breakdown':<35s}")
    for r in rows:
        strr = r["n_judged"] / r["n_total"] * 100 if r["n_total"] else 0
        types = ", ".join(f"{k}:{v}" for k, v in sorted(r["silence_types"].items()))
        mean_sil = r["mean_silence_appr"]
        mean_sil_s = f"{mean_sil:.2f}" if mean_sil is not None else "—"
        print(f"{r['name']:30s} {r['n_total']:>4d} {strr:>5.1f}% {r['n_silent']:>5d} {mean_sil_s:>9s}   {types:<35s}")

    print(f"\n{'='*100}")
    print(f"Per-dimension means (3 views)")
    print(f"{'='*100}")
    for view_name, key in [
        ("JUDGED-ONLY (silent excluded)", "judged_only"),
        ("SILENCE = 0 (lower bound)", "silence_as_0"),
        ("SILENCE = JUDGED (1-5 by GPT)", "silence_judged"),
    ]:
        print(f"\n[{view_name}]")
        header = " ".join(f"{d[:10]:>10s}" for d in DIMS)
        print(f"{'model':30s} {header}")
        for r in rows:
            vals = " ".join(
                (f"{r[key][d]:>10.3f}" if r[key].get(d) is not None else f"{'—':>10s}")
                for d in DIMS
            )
            print(f"{r['name']:30s} {vals}")


if __name__ == "__main__":
    main()
