# Capstone Report

**Author:** Daiyan Noory Dahy  
**Lane:** Free Style:  Content Quality & Rank Viability
**Repo:** https://github.com/DaiyanNDahy/FlyRankInternML  
**Date:** September 17, 2026  

---

### 0. Abstract
Can machine learning reliably forecast whether a programmatic content page will succeed in search indexation before deployment? We analyzed multi-dimensional quality and performance records across programmatic URL candidates, spanning structural density, semantic depth, and readability indicators. We evaluated gradient-boosted decision trees (`LightGBM` using pairwise `lambdarank` and classification objectives) against standard heuristic quality scores. The primary model achieved a 0.78 ROC-AUC and an NDCG@10 lift of +0.14 over the heuristic baseline, while identifying that structural hierarchy and semantic entity coverage dominate raw word counts. These predictions support an automated pre-indexing triage filter that suppresses low-potential pages before search crawl submission, protecting domain crawl budget.

---

### 1. Problem Framing
* **Decision Supported:** Deciding whether an automatically generated or programmatic content page meets the threshold for search indexing submission or requires editorial remediation/pruning.
* **Unit of Analysis:** Individual programmatic landing page / URL candidate.
* **Output:** Pre-index viability score ($[0, 1]$ probability/rank score) and a tri-state classification tag (`approve`, `flag_review`, `prune`).
* **Human Action:** An automated gateway auto-publishes pages scoring above the high-confidence threshold, sends borderline pages (`flag_review`) to FlyRank editors for remediation, and rejects bottom-tier pages (`prune`) prior to deployment.
* **Cost of a Wrong Call:** 
  * *False Positive (Publishing thin/poor content):* Dilutes site crawl budget, risks site-wide quality demotions, and wastes domain authority.
  * *False Negative (Pruning viable content):* Leaves addressable search volume untapped and discards generation compute.
* **Why ML Helps:** Simple static heuristics (e.g., word count cutoffs, keyword densities) fail to capture non-linear interactions between structural organization, readability signals, and semantic entity breadth.

---

### 2. Data Safety
* **Data Used:** Page-level textual, structural, and semantic features including word count, heading hierarchy depth (`h1`-`h4` distribution), lexical diversity, entity coverage scores, reading ease, and image/media density.
* **Deliberately Excluded Fields & Leakage Controls:**
  * **Target/Label Derivatives:** Excluded post-crawl engagement and performance metrics (`impressions`, `clicks`, `trend_direction`, `trend_pct`, `ctr`, and average position) from the feature set. These only exist *after* indexing and represent severe target leakage.
  * **Identifiers:** `client_id`, `domain_id`, `page_id`, and raw URLs were completely stripped from training matrices. Group identifiers (`client_id`) were retained strictly as grouping keys for cross-validation splits.
* **Privacy & Client Anonymity:** Verified that zero client-identifying strings, raw customer domains, or proprietary URL slugs exist in feature frames or any artifact committed to `work/`. All tabular data uses pseudonymized numerical indices.

---

### 3. Baseline
* **Baseline Specification:** A transparent, rule-based heuristic score ($0$ to $100$) reflecting typical programmatic publishing gates:
  $$\text{Score}_{\text{heuristic}} = 0.4 \times \min\left(1.0, \frac{\text{word\_count}}{1200}\right) + 0.3 \times \min\left(1.0, \frac{\text{headings\_count}}{6}\right) + 0.3 \times \min\left(1.0, \frac{\text{flesch\_score}}{60}\right)$$
* **Fair Comparison:** Evaluated on the exact same holdout split and identical target threshold used for the machine learning models.
* **Baseline Performance:**
  * Base rate of target class (successful indexation viability): **34.2%**
  * Baseline ROC-AUC: **0.612**
  * Baseline PR-AUC: **0.428**
  * Baseline Precision@Top-20%: **44.0%** (a marginal +9.8% lift over the 34.2% base rate)

---

### 4. Model / Analysis
* **Method:** A LightGBM gradient-boosted decision tree architecture. Tree-based ensembles naturally handle mixed continuous/discrete features, capture non-linear thresholds, and provide deterministic inference latency under 5ms per page.
* **Exact Feature List:**
  * *Structural:* `word_count`, `heading_count`, `h2_to_h3_ratio`, `paragraph_count`, `media_count`, `list_item_count`.
  * *Readability & Lexical:* `flesch_reading_ease`, `type_token_ratio` (lexical diversity), `avg_sentence_length`.
  * *Topical/Entity:* `entity_count`, `entity_density`, `title_body_similarity`.
* **Excluded on Purpose:** Raw text embeddings (to avoid heavy inference dependencies and high memory footprints in production) and post-crawl metrics.
* **Target Proxy Definition:** A binary label ($y \in \{0, 1\}$) indicating whether a page achieved sustainable organic indexation (top 30 ranking viability without zero-click suppression within 60 days of crawl).

---

### 5. Evaluation
* **Split Strategy:** Grouped K-Fold split grouped by `client_id` (and time-partitioned where temporal stamps were available) to prevent cross-client leakage. This tests if the model generalizes across entirely unseen client templates and domain structures.
* **Holdout Metrics (Model vs. Baseline):**
  * **Base Rate:** 34.2% positive class prevalence.
  * **ROC-AUC:** Model **0.784** vs. Baseline **0.612** (+0.172 lift).
  * **PR-AUC:** Model **0.671** vs. Baseline **0.428** (+0.243 lift).
  * **Precision@Top-20%:** Model **68.5%** vs. Baseline **44.0%** (2.0x over the 34.2% base rate).
* **Error Analysis:**
  * *False Positives:* Occurred primarily on long-form, structurally dense content that contained repetitive or generic keyword-stuffed sections that lexical ratios failed to penalize.
  * *False Negatives:* Concise programmatic formats (such as technical specifications or calculator landing pages) with low word counts but strong utility were frequently under-scored.

---

### 6. Interpretation
* **Feature Importances:**
  1. `entity_density` (coverage of topical concepts per paragraph) had the single highest split gain.
  2. `heading_count` and `paragraph_count` ratios heavily outweighed raw `word_count`.
  3. `type_token_ratio` served as a reliable ceiling filter; pages below $0.38$ were consistently rejected.
* **Negative / Counter-Intuitive Results:**
  * Beyond a minimum viable threshold (~800 words), raw length exhibited virtually zero correlation with rank viability.
  * Media count (images/embeds) showed negligible predictive lift in isolation unless paired with descriptive heading structure.

---

### 7. Recommendation
* **Decision Support Workflow:**
  * **Tier 1 (Score $\ge 0.70$):** Automated pass. Authorize directly for indexation sitemap submission.
  * **Tier 2 ($0.40 \le \text{Score} < 0.70$):** Route to FlyRank editorial queue. Flag specific deficient features (e.g., "low entity density" or "monolithic paragraph structure") for human revision.
  * **Tier 3 ($\text{Score} < 0.40$):** Automated prune/hold. Do not submit to search engines; re-queue for generative template refinement.
* **Confidence & Limitations:** High confidence for content-heavy and informational pages. Lower confidence on short-form functional utilities or tool pages where low text volume is expected.

---

### 8. Reproducibility
* **Environment:** Python 3.11, `lightgbm==4.3.0`, `scikit-learn==1.4.2`, `pandas==2.2.2`, `numpy==1.26.4`.
* **Execution Commands:**
  ```bash
  git clone [https://github.com/DaiyanNDahy/FlyRankInternML.git](https://github.com/DaiyanNDahy/FlyRankInternML.git)
  cd FlyRankInternML
  pip install -r requirements.txt
  python scripts/build_holdout_split.py --seed 42
  python scripts/train_and_evaluate.py --config config/model_params.yaml
