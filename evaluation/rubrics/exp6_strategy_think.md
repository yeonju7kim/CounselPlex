# Exp 6 — Strategy & Think Analysis

## Strategy Analysis (annotation-based, external classifier)

Strategy is not predicted by the model — it is an annotation from the dataset.
To measure whether the model implicitly follows appropriate strategies,
classify the model's transcribed response with an LLM and compare to gt_strategy.

### LLM-based Strategy Classification
```python
# Prompt GPT-4o to classify the response into one of 14 strategies
# Compare with gt_strategy annotation
from sklearn.metrics import classification_report
# llm_predicted_strategy vs. gt_strategy across all test turns
```

### Strategy Diversity (entropy)
```python
from scipy.stats import entropy
counts = Counter(llm_predicted_strategies)
probs = [c / total for c in counts.values()]
diversity = entropy(probs)
```
High diversity → model uses varied strategies rather than collapsing to one.

---

## Think–Response Consistency

Does the Think content actually drive the response? Two methods:

### Cosine Similarity (fast)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
think_emb = model.encode(think_text)
response_emb = model.encode(response_text)
sim = cosine_similarity(think_emb, response_emb)
```

### NLI Entailment (more precise)
```python
# Does think_text entail / support response_text?
# Use a NLI model (e.g., cross-encoder/nli-deberta-v3-small)
# Hypothesis: response_text  Premise: think_text
# Score: P(entailment)
```

Expected: our model (w/ CoT) should have higher think–response consistency than no-CoT baseline.

---

## Think Diversity

Are think texts diverse, or does the model repeat generic phrases?

```python
# Distinct-n over think_text corpus
def distinct_n(texts, n):
    ngrams = [ng for t in texts for ng in zip(*[t.split()[i:] for i in range(n)])]
    return len(set(ngrams)) / len(ngrams)

distinct_1 = distinct_n(think_texts, 1)
distinct_2 = distinct_n(think_texts, 2)
```
