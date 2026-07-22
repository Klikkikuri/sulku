# Generalist AI Identification Model: Data & Evaluation Architecture

## Overview

The Tier 2 Generalist Model serves as a fallback classifier when the Tier 1 specialist ensemble produces inconclusive results ($Z_{\text{combined}} < Z_{\text{threshold}}$). Unlike specialist models tuned to recognize specific generator families (e.g. GPT-style, Claude-style, Llama-style), the generalist classifier must detect broad, generator-agnostic structural and stylistic markers of machine-generated text.

This document details the critical data considerations, validation methodologies, and contamination risks that determine whether a generalist identification model successfully generalizes to unseen AI generators.

---

## Key Data & Experimental Considerations

### 1. Stratified Generator Balancing

#### The Risk
If the AI training pool is imbalanced (for example, containing 10,000 GPT-4 samples but only 1,000 DeepSeek or Mistral samples), the model will optimize its decision boundary for GPT-4 features. Instead of becoming generator-agnostic, it turns into a diluted GPT-specialist bearing a generalist label.

#### Mitigation Strategy
- **Explicit Stratification**: Dataset assembly must strictly enforce equal sample volume across all known generator families $G = \{G_1, G_2, \dots, G_m\}$:
  $$N(G_1) = N(G_2) = \dots = N(G_m) = N$$
- **Balanced Class Ratios**: The total AI training pool of $m \times N$ samples is matched against an equal volume of $m \times N$ human samples.

---

### 2. Leave-One-Generator-Out (LOGO) Cross-Validation

#### The Risk
Standard random train/test splits overestimate model performance because test samples originate from generator families present during training. Evaluating a generalist on known generator distributions measures memorization, not generalization.

#### Validation Protocol
To prove the claim *"this model detects AI text from generators it was never trained on"*, we enforce **Leave-One-Generator-Out (LOGO)** cross-validation:

1. For each generator family $G_k \in G$:
   - Train the generalist on data from $G \setminus \{G_k\}$.
   - Evaluate exclusively on test samples from held-out generator $G_k$.
2. Measure performance metrics ($\text{LOGO-AUC}_k$, $\text{LOGO-F1}_k$).
3. Certification Rule: The generalist model is certified for deployment only if:
   $$\min_k \text{LOGO-AUC}_k \ge 0.85$$

If performance drops sharply on held-out generator $G_k$, the model has failed to generalize and functions merely as an ensemble with an extra vote.

---

### 3. Topic & Prompt Leakage Prevention

#### The Risk
When human and AI samples are generated from mismatched prompt sets or distinct topical domains, the classifier can learn topic artifacts (e.g. specific vocabulary, bulleted summaries, or AI response conventions) rather than true "AI-ness". This produces deceptively high validation accuracy that collapses on real-world input.

#### Mitigation Strategy
- **Paired Corpus Construction**: AI samples should be generated using prompts derived directly from the human text corpus (e.g. YLE news articles as human text paired with AI-generated text produced from YLE article headlines/prompts).
- **Domain-Adversarial Evaluation**: Test splits must include independent topic distributions to verify that classification performance remains invariant across subject matter.

---

### 4. Human Corpus Contamination & Distillation Correlation

#### The Risk
Web-crawled "human-written" text increasingly contains AI-assisted edits (such as proofreading, LLM rephrasing, or automated summarization). Contamination in the human training set creates a fuzzy ground-truth boundary. This undermines a generalist model far more severely than a narrow specialist, as specialists check for one specific generator's distinct style, whereas generalists rely on sharp global boundaries.

#### Mitigation Strategy
- **Temporal Provenance Filtering**: Utilize verified pre-2022 human text archives (prior to the mass adoption of modern LLM editing tools).
- **Verified Editorial Sources**: Restrict human baseline corpora to curated, editorially verified collections (e.g., YLE news archives with documented human editorial pipelines).
- **Exclusion of Hybrid Text**: Exclude mixed AI/human edited documents from training splits to maintain uncorrupted ground truth.

---

## Summary Matrix

| Domain | Risk | Technical Solution | Validation Standard |
| :--- | :--- | :--- | :--- |
| **Data Balance** | Dominant generator bias (e.g. 10x GPT sample dominance). | Explicit per-generator stratification ($N_i = N$). | Verified equal sample distribution across all $m$ generators. |
| **Generalization** | Overfitting to known generator distributions. | Leave-One-Generator-Out (LOGO) cross-validation. | $\min_k \text{LOGO-AUC}_k \ge 0.85$ on held-out generators. |
| **Topic Leakage** | Learning vocabulary/topic shortcuts instead of style. | Paired prompt/human corpus pairing & adversarial domain tests. | Stable AUC across novel topic domains. |
| **Corpus Purity** | AI-edited human text corrupting ground-truth boundaries. | Pre-2022 temporal filtering & verified editorial sources (YLE). | Zero LLM-assisted passages in human training baseline. |
