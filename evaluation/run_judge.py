"""
GPT-4o judge for content-quality eval (Exp 1).

Reads dialogue results from Results/results/{model_dir}/, scores each
non-silent supporter turn with GPT-4o using the offline rubric in
Evalutation/markdown/exp1_content_quality.md, and writes per-dialogue
eval JSONs to Results/eval/{model_dir}/.

Usage:
  python Evalutation/run_judge.py
  python Evalutation/run_judge.py --max-dialogues 10
  python Evalutation/run_judge.py --results-dir Results/results/original \
      --out-dir Results/eval/original
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # CounselPlex/
# OpenAI API key is read from the OPENAI_API_KEY environment variable.

ESCOT_14 = [
    "Question", "Restatement_or_Paraphrasing", "Reflection_of_Feelings",
    "Self-disclosure", "Affirmation_and_Reassurance", "Providing_Suggestions",
    "Information", "Specify", "Summarize", "Imagery", "Immediacy",
    "Take_Responsibility", "Homework_Assignments", "Others",
]

ESCOT_TO_ESCONV_8 = {
    "Question": "Question",
    "Restatement_or_Paraphrasing": "Restatement_or_Paraphrasing",
    "Reflection_of_Feelings": "Reflection_of_Feelings",
    "Self-disclosure": "Self-disclosure",
    "Affirmation_and_Reassurance": "Affirmation_and_Reassurance",
    "Providing_Suggestions": "Providing_Suggestions",
    "Information": "Information",
    "Others": "Others",
    "Specify": "Question",
    "Summarize": "Restatement_or_Paraphrasing",
    "Imagery": "Providing_Suggestions",
    "Immediacy": "Reflection_of_Feelings",
    "Take_Responsibility": "Self-disclosure",
    "Homework_Assignments": "Providing_Suggestions",
}

JUDGE_PROMPT_VERSION = "exp1_v4_1_qa_coherence"

JUDGE_SYSTEM = "You are an expert counseling-dialogue evaluator. Return JSON only."

JUDGE_USER_TEMPLATE = """\
IMPORTANT — OFFLINE EVALUATION:
- The supporter response below replaces the original supporter turn in a dataset.
- The seeker has NOT seen this response. Any later seeker utterance you can imagine
  would NOT be a reaction to this response.
- Evaluate ONLY based on (a) the prior conversation history, (b) the seeker's
  current utterance, and (c) the supporter response itself.
- Do NOT speculate about how the seeker would react.

[Conversation history]
{history}

[Current turn]
Seeker: {seeker_text}
Supporter: {response_text}

Rate the supporter's response on each dimension using this 1-5 rubric:
  1 = clear failure (broken / non-sequitur / harmful / contradictory)
  2 = poor (off-topic, generic platitude, ignores key elements)
  3 = adequate (relevant and appropriate but generic or surface-level)
  4 = good (specific, attuned, addresses the seeker's actual concern)
  5 = excellent (highly attuned, specific, demonstrates deep skill)

For scores 4-5, provide concrete justification in `reason`.
For scores 1-2, describe what is missing or wrong.

Dimensions:

- empathy: emotionally supportive, appropriate to the seeker's distress.
  - High (4-5): names/validates seeker's specific emotion, conveys warmth, concrete emotional attunement.
  - Mid (3): polite acknowledgment without specific engagement. Generic clichés
    ("I'm sorry to hear that", "That must be hard") without specificity default to 3
    or LOWER, even if grammatically polite.
  - Low (1-2): cold, dismissive, ignores emotional content, OR clichéd-only response to severe distress.

- coherence (Q-A coherence): does the supporter response directly engage with the
  seeker's MOST RECENT utterance?
  - High (4-5): directly addresses seeker's question/disclosure/emotion with relevant content.
  - Mid (3): relevant but partial or generic deflection.
  - Low (1-2): non-sequitur, ignores the seeker, or generic phrase unrelated to what the seeker just said.
    (Example: seeker discloses death of a parent → supporter says "Would you like to talk about it?"
    scores 3 at best, NOT high.)
  - ⚠ A response can be perfectly grammatical and still score 1-2 here. Grammar is NOT coherence.
    If supporter ignores or sidesteps what the seeker just said, score 1-2 regardless of fluency.

- context_recalling (whole-dialogue context): correctly uses information accumulated
  across earlier turns — facts, emotions, situational context.
  - 5 (excellent): integrates MULTIPLE threads from earlier turns (e.g. job loss + addiction +
    guilt toward family) into a single attuned response.
  - 4 (good): explicitly references one earlier disclosed detail.
  - 3 (adequate): consistent with prior turns but does not reference them.
  - 1-2: contradicts earlier facts, forgets emotional state, or misattributes information.
  - NOTE: the supporter inventing a plausible self-disclosure from their own "life" is a
    recognized counseling strategy — do NOT penalize as a recall failure.

- strategy_appropriateness: is the strategy used appropriate for the current emotional context?
  - High (4-5): strategy is one of the best choices for this moment.
  - Mid (3): a reasonable but not optimal strategy.
  - Low (1-2): strategy mismatches the seeker's need (e.g. premature suggestion when seeker
    is venting and needs validation).

- stressor_awareness: does the response engage with the seeker's CORE STRESSOR
  (the concrete underlying SITUATION causing distress), as opposed to drifting
  toward surface emotional symptoms (sadness, anxiety, sleeplessness)?
  - First, INFER the core stressor from the FULL dialogue context — the
    underlying actionable situation (e.g. "lost childhood friend to covid",
    "job loss with bills piling up", "abusive partner"). Symptoms like
    "sleepless nights" or "feeling sad" are NOT the stressor — they are
    consequences of it.
  - 5 (excellent): directly engages with the core stressor in a way that helps
    the seeker process the root (e.g. for grief: acknowledges the loss
    specifically; for job loss: addresses career uncertainty meaningfully).
  - 4 (good): clearly aware of the core stressor — references it or anchors
    the response within it, even if the immediate suggestion is about a symptom.
  - 3 (adequate): consistent with the stressor but does not engage it; stays
    on surface emotional content without contradicting context.
  - 2 (poor): treats a SYMPTOM as if it were the problem (e.g. for a grieving
    seeker with insomnia, responds purely with sleep-hygiene tips and ignores
    the loss).
  - 1 (off): completely misidentifies the stressor, contradicts it, or
    addresses an unrelated issue as primary.
  - NOTE: when no clear external stressor exists in the conversation (e.g.
    light check-in, only emotional state revealed without concrete situation),
    score 3 by default — the response cannot fail to engage what is not there.

Strategy classification — select ALL strategies that clearly apply, ORDERED BY PRIMACY
(most central to the response first). Choose 1 to 3 labels. Counseling responses often
combine strategies (e.g. validation + question, self-disclosure + suggestion).

Available labels (ESCoT 14):
[Question, Restatement_or_Paraphrasing, Reflection_of_Feelings, Self-disclosure,
 Affirmation_and_Reassurance, Providing_Suggestions, Information,
 Specify, Summarize, Imagery, Immediacy, Take_Responsibility, Homework_Assignments,
 Others]

Brief notes:
- Question: open question to understand. Specify: a question for narrow specific details.
- Restatement_or_Paraphrasing: rephrasing seeker's recent utterance. Summarize: condensing multiple turns.
- Reflection_of_Feelings: naming the seeker's emotion. Affirmation_and_Reassurance: validating, comforting.
- Self-disclosure: supporter shares own experience. Immediacy: discussing the here-and-now relationship.
- Providing_Suggestions: general advice. Homework_Assignments: concrete take-home task.
- Imagery: inviting visualization. Information: factual knowledge/resources (not advice).
- Take_Responsibility: emphasizing seeker's agency. Others: nothing else fits.

Avoid over-labeling: include a label only when it is genuinely present, not merely implied.
Use Others only when no other label fits.

Also output a SEPARATE primary_strategy field as a sanity check:
- primary_strategy: the SINGLE most central strategy (one label).
  This MUST equal predicted_strategies[0]. We verify consistency.

Return JSON ONLY with this exact schema:
{{
  "empathy": 1|2|3|4|5,
  "coherence": 1|2|3|4|5,
  "context_recalling": 1|2|3|4|5,
  "strategy_appropriateness": 1|2|3|4|5,
  "stressor_awareness": 1|2|3|4|5,
  "predicted_strategies": ["<primary label>", "<secondary label>", ...],
  "primary_strategy": "<the single primary label>",
  "reason": "<concrete one-sentence justification — required for scores 4-5 and 1-2>"
}}
"""


def normalize_label(s):
    if not s:
        return ""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def build_history(prev_turns):
    """Render prior turns as 'Seeker: ...\\nSupporter: ...' using GT context."""
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


def call_judge(client, model, history, seeker_text, response_text, max_retries=3):
    user = JUDGE_USER_TEMPLATE.format(
        history=history,
        seeker_text=seeker_text,
        response_text=response_text,
    )
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
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"judge call failed after {max_retries} retries: {last_err}")


def evaluate_dialogue(client, judge_model, dialogue):
    conv_id = dialogue["conv_id"]
    turns = dialogue["turns"]
    out_turns = []

    for i, t in enumerate(turns):
        turn_id = t["turn_id"]
        gt_strategy = t.get("strategy", "")
        seeker = (t.get("seeker_gt") or "").strip()
        response = (t.get("supporter_gen_speech") or "").strip()

        if not response:
            out_turns.append({
                "turn_id": turn_id,
                "engagement": 0,
                "supporter_response": "",
                "gt_strategy": gt_strategy,
                "skipped": "no_response",
            })
            continue

        history = build_history(turns[:i])
        try:
            judged = call_judge(client, judge_model, history, seeker, response)
        except Exception as e:
            out_turns.append({
                "turn_id": turn_id,
                "engagement": 1,
                "supporter_response": response,
                "gt_strategy": gt_strategy,
                "error": str(e),
            })
            continue

        # Multi-label predicted strategies: list, ordered by primacy.
        raw_list = judged.get("predicted_strategies") or []
        if isinstance(raw_list, str):
            raw_list = [raw_list]
        pred14_list = []
        for raw in raw_list:
            n = normalize_label(raw)
            canon = next((s for s in ESCOT_14 if normalize_label(s) == n), None)
            if canon and canon not in pred14_list:
                pred14_list.append(canon)
        if not pred14_list:
            pred14_list = ["Others"]
        pred8_list = []
        for c in pred14_list:
            mapped = ESCOT_TO_ESCONV_8.get(c, "Others")
            if mapped not in pred8_list:
                pred8_list.append(mapped)

        # Independent primary field (for sanity check on ordering)
        primary_raw = judged.get("primary_strategy", "")
        primary_norm = normalize_label(primary_raw)
        primary_canon = next((s for s in ESCOT_14 if normalize_label(s) == primary_norm), None)
        primary_consistent = (primary_canon == pred14_list[0]) if primary_canon else None

        out_turns.append({
            "turn_id": turn_id,
            "engagement": 1,
            "supporter_response": response,
            "gt_strategy": gt_strategy,
            "predicted_strategies_14": pred14_list,
            "predicted_strategies_8_mapped": pred8_list,
            "predicted_strategy_14": pred14_list[0],   # primary = list[0]
            "predicted_strategy_8_mapped": pred8_list[0],
            "primary_strategy_separate": primary_canon,
            "primary_consistent_with_list_0": primary_consistent,
            "empathy": int(judged.get("empathy", 0)),
            "coherence": int(judged.get("coherence", 0)),
            "context_recalling": int(judged.get("context_recalling", 0)),
            "strategy_appropriateness": int(judged.get("strategy_appropriateness", 0)),
            "stressor_awareness": int(judged.get("stressor_awareness", 0)),
            "reason": judged.get("reason", ""),
        })

    return {
        "conv_id": conv_id,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "turns": out_turns,
        "summary": summarize(out_turns),
    }


def summarize(out_turns):
    n_turns = len(out_turns)
    judged = [t for t in out_turns if t.get("engagement") == 1 and "error" not in t and "empathy" in t]
    n_responded = sum(1 for t in out_turns if t.get("engagement") == 1)
    strr = n_responded / n_turns if n_turns else 0.0

    def mean(key):
        vals = [t[key] for t in judged if key in t]
        return sum(vals) / len(vals) if vals else None

    # Multi-label distributions: count each label per turn (a turn with [Question, Self-disclosure]
    # contributes +1 to each).
    dist14 = Counter()
    dist8 = Counter()
    for t in judged:
        for s in t.get("predicted_strategies_14", [t.get("predicted_strategy_14")]):
            if s: dist14[s] += 1
        for s in t.get("predicted_strategies_8_mapped", [t.get("predicted_strategy_8_mapped")]):
            if s: dist8[s] += 1

    return {
        "n_turns": n_turns,
        "n_responded": n_responded,
        "n_judged": len(judged),
        "strr": round(strr, 3),
        "no_response_rate": round(1 - strr, 3),
        "empathy_mean": round(mean("empathy"), 3) if mean("empathy") is not None else None,
        "coherence_mean": round(mean("coherence"), 3) if mean("coherence") is not None else None,
        "context_recalling_mean": round(mean("context_recalling"), 3) if mean("context_recalling") is not None else None,
        "strategy_appropriateness_mean": round(mean("strategy_appropriateness"), 3) if mean("strategy_appropriateness") is not None else None,
        "stressor_awareness_mean": round(mean("stressor_awareness"), 3) if mean("stressor_awareness") is not None else None,
        "strategy_distribution_14": dict(dist14),
        "strategy_distribution_8_mapped": dict(dist8),
    }


def aggregate_print(all_evals):
    n = len(all_evals)
    print(f"\n{'=' * 60}\nAggregate over {n} dialogues\n{'=' * 60}")

    def pooled_mean(metric):
        vals = []
        for e in all_evals:
            for t in e["turns"]:
                if metric in t and t.get("engagement") == 1:
                    vals.append(t[metric])
        if not vals:
            return None, 0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return (m, var ** 0.5, len(vals))

    total_turns = sum(e["summary"]["n_turns"] for e in all_evals)
    total_resp = sum(e["summary"]["n_responded"] for e in all_evals)
    print(f"Total turns: {total_turns} | Responded: {total_resp} | "
          f"STRR: {total_resp / total_turns:.3f} | No-response: {1 - total_resp / total_turns:.3f}")

    for k in ("empathy", "coherence", "context_recalling", "strategy_appropriateness", "stressor_awareness"):
        res = pooled_mean(k)
        if res[0] is not None:
            m, sd, n = res
            print(f"{k:26s}: {m:.3f} ± {sd:.3f}  (n={n})")

    dist14 = Counter()
    n_judged = 0
    for e in all_evals:
        for t in e["turns"]:
            if t.get("engagement") == 1 and "predicted_strategies_14" in t:
                n_judged += 1
                for s in t["predicted_strategies_14"]:
                    dist14[s] += 1
            elif "predicted_strategy_14" in t:
                n_judged += 1
                dist14[t["predicted_strategy_14"]] += 1
    print(f"\nStrategy distribution (ESCoT-14, multi-label across {n_judged} turns):")
    for s, c in dist14.most_common():
        print(f"  {s:30s} {c:4d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="dir of *.result.json files (output of build_eval_results.py)")
    p.add_argument("--out-dir", required=True,
                   help="dir to write per-dialogue *.eval.json")
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

    files = sorted(f for f in results_dir.glob("*.result.json"))
    if args.max_dialogues is not None:
        files = files[: args.max_dialogues]
    print(f"Found {len(files)} dialogue file(s). Output → {out_dir}")

    client = OpenAI(api_key=api_key)
    all_evals = []
    for f in files:
        conv_id = f.name.replace(".result.json", "")
        out_path = out_dir / f"{conv_id}.eval.json"
        if out_path.exists():
            print(f"[skip] {conv_id} (exists)")
            all_evals.append(json.loads(out_path.read_text()))
            continue

        print(f"[eval] {conv_id} ...", flush=True)
        dialogue = json.loads(f.read_text())
        ev = evaluate_dialogue(client, args.judge_model, dialogue)
        ev["model_dir"] = results_dir.name
        out_path.write_text(json.dumps(ev, indent=2, ensure_ascii=False))
        s = ev["summary"]
        print(f"   STRR={s['strr']}  empathy={s['empathy_mean']}  "
              f"coh={s['coherence_mean']}  recall={s['context_recalling_mean']}  "
              f"strat_appr={s['strategy_appropriateness_mean']}  "
              f"stressor={s['stressor_awareness_mean']}")
        all_evals.append(ev)

    aggregate_print(all_evals)


if __name__ == "__main__":
    main()
