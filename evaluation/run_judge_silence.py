"""
GPT-4.1 judge for silence-appropriateness.

For each turn where the supporter did NOT speak (empty supporter_gen_speech),
score on 1-5 how appropriate that silence was given the conversation context.
Distinguishes harmful silence (avoidance / dropped response in moment of need)
from therapeutic silence (giving space after disclosure / minimal response).

Outputs Results/eval_silence/{model_dir}/{conv_id}.silence.json.

Usage:
  python Evalutation/run_judge_silence.py \
      --results-dir Results/results/2026-05-14_cot_v3_strat/epoch_001_greedy \
      --out-dir Results/eval_silence/2026-05-14_cot_v3_strat/epoch_001_greedy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # CounselPlex/
# OpenAI API key is read from the OPENAI_API_KEY environment variable.

JUDGE_PROMPT_VERSION = "silence_v1"

JUDGE_SYSTEM = (
    "You are an expert counseling-dialogue evaluator. The supporter said NOTHING this turn. "
    "Decide whether that silence was appropriate. Return JSON only."
)

JUDGE_USER_TEMPLATE = """\
IMPORTANT — OFFLINE EVALUATION OF SUPPORTER SILENCE:
- At this turn position the supporter did NOT speak (no response, no minimal acknowledgement).
- Evaluate whether silence was appropriate, given the conversation history and the seeker's
  current utterance.
- Do NOT speculate about later seeker reactions.

[Conversation history]
{history}

[Current turn]
Seeker: {seeker_text}
Supporter: (silent — no response)

Rate the appropriateness of this silence on a 1-5 scale:

  1 = harmful silence. Seeker is in acute distress, asked a direct question, just disclosed
      something heavy, or is clearly waiting for engagement — silence here is avoidance /
      dropped response.
  2 = poor silence. Seeker had a clear opening for support; silence is a missed opportunity.
  3 = neutral silence. Silence neither clearly helps nor hurts; seeker can continue without
      damage but support could have added value.
  4 = good silence. Appropriate space — seeker may benefit from time to process. Minimal
      response would also fit, but silence is acceptable.
  5 = excellent silence. This is a textbook therapeutic moment to hold space — e.g., right
      after a heavy disclosure / sustained emotional release, where any words would intrude.

Also classify the silence type:
  - "avoidance"  : harmful silence in a moment that demanded engagement
  - "neutral"    : silence with no clear pedagogical role
  - "therapeutic": intentional pause that gives space appropriate to the moment

Return JSON ONLY with this exact schema:
{{
  "silence_appropriateness": 1|2|3|4|5,
  "silence_type": "avoidance" | "neutral" | "therapeutic",
  "reason": "<one-sentence justification>"
}}
"""


def build_history(prev_turns):
    if not prev_turns:
        return "(none — this is the first supporter turn)"
    lines = []
    for t in prev_turns:
        seeker = (t.get("seeker_gt") or "").strip()
        supporter = (t.get("supporter_gt") or "").strip()
        if seeker:
            lines.append(f"Seeker: {seeker}")
        if supporter:
            lines.append(f"Supporter: {supporter}")
    return "\n".join(lines) if lines else "(empty)"


def call_judge(client, model, history, seeker_text, max_retries=3):
    user = JUDGE_USER_TEMPLATE.format(history=history, seeker_text=seeker_text)
    last_err = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} (sleep {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"silence-judge failed after {max_retries} retries: {last_err}")


def evaluate_dialogue(client, judge_model, dialogue):
    conv_id = dialogue["conv_id"]
    turns = dialogue["turns"]
    out_turns = []
    n_silent = 0
    for i, t in enumerate(turns):
        resp = (t.get("supporter_gen_speech") or "").strip()
        if resp:
            continue
        n_silent += 1
        history = build_history(turns[:i])
        seeker = (t.get("seeker_gt") or "").strip()
        try:
            j = call_judge(client, judge_model, history, seeker)
        except Exception as e:
            out_turns.append({"turn_id": t["turn_id"], "error": str(e)})
            continue
        out_turns.append({
            "turn_id": t["turn_id"],
            "silence_appropriateness": int(j.get("silence_appropriateness", 0)),
            "silence_type": j.get("silence_type", ""),
            "reason": j.get("reason", ""),
        })
    return {
        "conv_id": conv_id,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "n_silent": n_silent,
        "silence_turns": out_turns,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-dialogues", type=int, default=None,
                   help="cap on dialogues to judge (None = all)")
    p.add_argument("--judge-model", default="gpt-4.1")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY environment variable not set")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("*.result.json"))
    if args.max_dialogues is not None:
        files = files[: args.max_dialogues]
    print(f"[{results_dir}] {len(files)} dialogue files. Output -> {out_dir}")

    client = OpenAI(api_key=api_key)
    total_silent = 0
    for f in files:
        conv_id = f.name.replace(".result.json", "")
        out_path = out_dir / f"{conv_id}.silence.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            print(f"[skip] {conv_id} (n_silent={existing.get('n_silent', 0)})")
            total_silent += existing.get("n_silent", 0)
            continue
        dialogue = json.loads(f.read_text())
        ev = evaluate_dialogue(client, args.judge_model, dialogue)
        ev["results_dir"] = str(results_dir)
        out_path.write_text(json.dumps(ev, indent=2, ensure_ascii=False))
        total_silent += ev["n_silent"]
        scores = [t.get("silence_appropriateness") for t in ev["silence_turns"] if "silence_appropriateness" in t]
        mean = sum(scores) / len(scores) if scores else None
        types = [t.get("silence_type") for t in ev["silence_turns"]]
        print(f"[{conv_id}] silent={ev['n_silent']}  mean_appr={mean}  types={types}")
    print(f"\nTotal silent turns scored: {total_silent}")


if __name__ == "__main__":
    main()
