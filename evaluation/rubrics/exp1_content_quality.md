# Exp 1 — Content Quality (vs. Baselines)

## Goal

Compare response quality across models. Primary experiment.
Two contributions to validate: **Think (CoT)** and **Dialogue context management (DCM)**.

## Baselines

| Model | Type |
|-------|------|
| LLM (text-only) | Text, turn-based |
| Moshi | Speech, duplex, no fine-tune |
| PersonaPlex (vanilla) | Speech, duplex, no fine-tune |
| **PersonaPlex + CoT + DCM (ours)** | Speech, duplex, CoT-supervised, dialogue context management |

## Granularity (per-turn vs per-dialogue)

Raw scoring is **per-turn**; aggregation produces both turn-level and dialogue-level numbers.

| Metric | Score unit | Note |
|--------|-----------|------|
| Empathy, Coherence | per-turn | local quality |
| Context Recalling | per-turn | judge sees full prior history when scoring |
| Strategy Appropriateness, Strategy Fit | per-turn | per-turn judgment |
| Strategy Distribution | per-dialogue | needs the full sequence |
| Engagement (STRR, No-response, etc.) | per-dialogue | needs the full sequence |
| Holistic Helpfulness (optional) | per-dialogue | one judgment over whole conv (1–3) |

## Offline assumption

The model's response **replaces** the original ESConv supporter turn. The dataset's *next* seeker utterance is a reaction to the **original** supporter — not to the model's response. The judge must therefore evaluate only on:
- the conversation history **prior** to the current turn,
- the seeker's current utterance,
- and the model's response itself.

The judge must NOT use, peek at, or imagine the seeker's reaction. The prompt makes this explicit.

## Metrics

Each metric maps to a specific claim.

| Category | Metric | Method | Score | Judge | Applies | Validates |
|----------|--------|--------|-------|-------|---------|-----------|
| Basic | **Engagement** (STRR / No-response / Late / Premature-interrupt) | Breakdown from turn-level speech logs | rates (per-dialogue) | auto | duplex models | "model does not stay silent" |
| Core | **Context Recalling** | Judge: does it correctly use information from earlier turns? | 1–3 | H+L | all | context management |
| Core | **Strategy Appropriateness** | Judge: is the chosen strategy appropriate for the emotional context? | 1–3 | H+L | all | think outcome |
| Core (ours only) | **Strategy Fit** | think.strategy ↔ response.strategy | rate | LLM | ours | think process |
| Quality | **Empathy** | Judge: emotionally supportive? | 1–3 | H+L | all | non-regression |
| Quality | **Coherence** | Judge: no nonsense / fragments / role confusion? | 1–3 | H+L | all | safety net |
| Aux | Strategy Distribution | Strategy histogram + entropy | H(p) | auto | all | counters "only does Reflection" |

### Definitions

- **Engagement breakdown** (duplex models, per-dialogue):
  - **STRR**: fraction of expected response turns where the model actually spoke (headline)
  - **No-response rate** = 1 − STRR
  - **Late-response rate**: model did speak but only after a silence threshold (e.g., 2s)
  - **Premature-interrupt rate**: model spoke while the seeker was still speaking
  - Text-only baseline: all N/A.
- **Context Recalling**: does the response correctly use information from earlier (possibly distant) turns — facts, emotions, context. Inventing self-disclosure is NOT a recall failure (it is a recognized ESConv strategy).
- **Coherence**: sentence completeness; no repetition, fragments, or role confusion. Catches decoding-level failures.
- **Strategy Appropriateness**: judge directly rates 1–3 whether the chosen strategy is a *reasonable* choice for the current emotional context. Not a match against GT — counseling has no single correct strategy, so GT-match is a weak signal. GT strategy is stored only for post-hoc analysis.
- **Strategy Fit**: agreement between the strategy declared in the Think trace and the strategy classified from the actual response. Validates that Think is not vestigial (process metric).

### Strategy diversity is not used alone

Strategy distribution entropy is reported as an auxiliary statistic only. Diverse ≠ Correct, so the main claim is carried by **Strategy Appropriateness**.

## LLM Judge Prompt

Rubric anchors (apply to all 1–3 metrics):
- **1 = Fail**: clear problem (absent / wrong / contradictory / inappropriate).
- **2 = Acceptable**: no clear problem, no clear strength.
- **3 = Good**: clearly succeeds at the dimension.

```
You are an expert counseling-dialogue evaluator.

IMPORTANT — OFFLINE EVALUATION:
- The supporter response below replaces the original supporter turn in a dataset.
- The seeker has NOT seen this response. Any later seeker utterance you can imagine
  would NOT be a reaction to this response.
- Evaluate ONLY based on (a) the prior conversation history, (b) the seeker's
  current utterance, and (c) the supporter response itself.
- Do NOT speculate about how the seeker would react.

[Conversation history (multi-turn)]
{history}

Current turn:
Seeker: {seeker_text}
Supporter: {response_text}

Rate the supporter's response on each dimension using this 1–3 rubric:
1 = clear failure, 2 = acceptable, 3 = clearly good.

Dimensions:
- Empathy: emotionally supportive and appropriate to the seeker's distress.
- Coherence: well-formed; no broken sentences, repetition, role confusion, or nonsense.
- Context Recalling: correctly uses information from earlier turns (facts, emotions, context).
  Note: inventing plausible self-disclosure is a recognized counseling strategy, NOT a recall failure.
- Strategy Appropriateness: is the chosen strategy appropriate for the current emotional context?
  Multiple strategies may be valid; judge whether the chosen one is a reasonable choice.

Classify the strategy used. Choose exactly one of (ESCoT 14):
[Question, Restatement_or_Paraphrasing, Reflection_of_Feelings, Self-disclosure,
 Affirmation_and_Reassurance, Providing_Suggestions, Information,
 Specify, Summarize, Imagery, Immediacy, Take_Responsibility, Homework_Assignments,
 Others]

Return JSON only:
{
  "empathy": 1|2|3,
  "coherence": 1|2|3,
  "context_recalling": 1|2|3,
  "strategy_appropriateness": 1|2|3,
  "predicted_strategy": "<one label>",
  "reason": "<one sentence justifying any score below 3>"
}
```

**Strategy Fit** (ours only): compare `think.strategy` (from CoT trace) with `predicted_strategy`.

## Human Eval

Focus on dimensions where LLM judges are unreliable:
- **Empathy**, **Coherence**, **Context Recalling**, **Strategy Appropriateness** with the same 1–3 rubric.
- 50–100 conversation subset, ≥2 annotators, report IAA (Krippendorff α).

## Input

- `response_text` from `turn_{N}_parsed.json` (text channel of `output.json`)
- `think.strategy` from CoT trace (ours only)
- GT strategy from `turn_{N}.json` (post-hoc analysis only)
- Turn-level silence/speech log for STRR

## Output

`results/eval/{model}/{conv_id}.eval.json` (per-dialogue):
```json
{
  "conv_id": "esconv_0042",
  "model_dir": "original",
  "judge_model": "gpt-4o",
  "turns": [
    {
      "turn_id": 2,
      "engagement": 1,
      "supporter_response": "...",
      "gt_strategy": "Reflection of feelings",
      "predicted_strategy_14": "Reflection_of_Feelings",
      "predicted_strategy_8_mapped": "Reflection_of_Feelings",
      "empathy": 3,
      "coherence": 3,
      "context_recalling": 2,
      "strategy_appropriateness": 3,
      "reason": ""
    }
  ],
  "summary": { ... }
}
```

`engagement` = 1 if the model produced speech in an expected turn, else 0.
`think_strategy` / `strategy_fit` only present for ours.

## Aggregate

Per model:
- **Engagement** (duplex only, per-dialogue → mean over conversations):
  - STRR, No-response rate, Late-response rate, Premature-interrupt rate
- **Empathy / Coherence / Context Recalling / Strategy Appropriateness**: mean ± std (1–3), per-turn aggregated
- **Strategy Distribution**: per-dialogue histogram + entropy, then averaged across dialogues
- **Strategy Fit**: rate, per-turn (ours only)
- **Holistic Helpfulness** (optional): per-dialogue 1–3 mean

Significance: paired t-test across conversations for 1–3 metrics; McNemar for binary.

## Expected Results (paper narrative)

| Metric | Text LLM | Moshi | PersonaPlex | Ours (CoT+DCM) | Validates |
|--------|----------|-------|-------------|----------------|-----------|
| STRR | N/A | mid | mid | high | duplex stability |
| Context Recalling | mid | low | low | high | context management (DCM) |
| Strategy Appropriateness | mid | low | low | high | think outcome |
| Strategy Distribution | skewed | skewed (Reflection) | skewed | balanced | counters "only does Reflection" |
| Empathy | high | high | high | high | non-regression |
| Coherence | high | mid? | mid? | high | safety net passed |
| Strategy Fit | N/A | N/A | N/A | high | think process |
