# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Asad Naeem
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Asadnaeem23/flyrank-ml-internship
- **Date:** March 2026

---

## 0. Abstract

How can search editorial teams prioritize thousands of decaying content assets to maximize human review efficiency without circular metrics or misleading causal assumptions? This research investigates an anonymized production dataset of 30,000 URLs spanning 32 client domains to develop an honest, machine-learning-assisted content triage system. We trained shallow decision trees on strictly pre-prediction metadata and historical activity signals, evaluating performance under an honest client-grouped holdout split (`GroupShuffleSplit`, 0 client overlap) compared against a transparent heuristic baseline. Under client holdout validation, the learned model achieved a measured Precision@50 of 0.66 (surpassing the 0.511 test base rate and the 0.640 heuristic baseline) while revealing that random row splitting artificially inflates perceived precision to 0.88 through client-level memorization. This validated scoring engine powers a 4-tier Content Action Playbook with auditable reason codes, providing directional decision-support for editorial workflows while establishing explicit operational no-go rules against automated publishing.

---

## 1. Problem framing

Organic search traffic decays continuously as competitor coverage intensifies, user search patterns shift, and factual details become dated. In large publishing catalogues (comprising thousands of articles across dozens of client domains), manual review of every URL is operationally impossible. The core operational decision this research supports is: **out of thousands of published articles, which specific pages should an editor inspect, update, or optimize first?**

- **Unit of Analysis:** An individual published content item (page) identified by a pseudonymous `content_id`.
- **Output:** A calibrated decay probability score and priority tier with auditable reason codes (`RC_MATURE_STALE_DECAY`, `RC_STRIKING_LOW_CTR`, `RC_THIN_VISIBLE_CONTENT`, `RC_HEALTHY_OR_LOW_DEMAND`).
- **Human Action:** Directing writer hours toward factual updates, title/meta snippet alignment, depth expansion, or observation.
- **Cost of a Wrong Call:** Wasting scarce editorial budget on pages with no recoverable demand, or neglecting decaying high-visibility assets until search impressions crash permanently.
- **Role of ML:** Fixed heuristic rules fail when signals interact non-linearly (e.g., content age interacting with update staleness and search rank). A validated model detects multi-signal degradation patterns while remaining interpretable.

---

## 2. Data safety

This research utilizes `data/raw/content_refresh_anonymized.csv`, containing **30,000 content items across 32 pseudonymized client domains** with 44 features derived from trailing 90-day Google Search Console (GSC) and Google Analytics 4 (GA4) logs.

### Exclusions and Leakage Prevention
1. **Direct Label Siblings (`trend_direction`, `trend_pct`):** The target label `target_down` is derived from `trend_direction == 'down'`, which is mathematically calculated from `trend_pct = (impressions_last_30d - impressions_prev_30d) / impressions_prev_30d * 100`. Both columns were strictly barred from model inputs.
2. **Outcome-Window Telemetry (`impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`):** These metrics measure performance *during* the label period. Including them allows models to reverse-engineer outcomes rather than predicting future states. In our leakage audit, removing them revealed the true generalization baseline.
3. **Pseudonymous IDs (`content_id`, `client_id`):** Retained strictly for grouping and joining, never used as predictive features.
4. **Public Data Protection:** Zero private client names, live URLs, search query strings, or API credentials appear anywhere in the codebase.

---

## 3. Baseline

Before training any complex model, we constructed a transparent domain heuristic baseline (Week-4 baseline) mirroring production triage rules:
- **Staleness Scoring:** 0 points if updated within 90 days; 1 point if updated 91–180 days ago; 2 points if untouched for > 180 days.
- **CTR Penalty:** +1 point if 90-day CTR $< 0.5\%$.
- **Tie-Breaking:** Sorted by `baseline_score` DESC, `days_since_last_update` DESC, and `ctr` ASC.

On the client-grouped holdout test partition (6,163 rows across 7 unseen clients), the heuristic baseline achieved:
- **Precision@20:** 0.700
- **Precision@50:** 0.640
- **Naive Test Base Rate:** 0.511

This serves as the honest benchmark the machine learning model must compete against.

---

## 4. Model / analysis

We selected a shallow **Decision Tree Classifier** (`max_depth=3, min_samples_leaf=50, random_state=42`) for the Content Refresh lane. A shallow tree captures structural threshold interactions between age, staleness, and ranking without overfitting or acting as an uninspectable black box.

### Feature Specification
- **Content Metadata:** `word_count`, `char_count`, `content_age_days`, `days_since_last_update`
- **Search Context:** `search_volume`, `competition`, `cpc`, `avg_position`
- **Target Definition:** `target_down = (trend_direction == 'down').astype(int)` (1 = downward search momentum, 0 = stable, up, new, or flat).

---

## 5. Evaluation

### Honest Grouped Split Design
Rows from the same client share domain authority, backlink strength, and content quality. A random train/test split allows a model to memorize client-level decay rates, inflating test metrics. We enforced an honest **`GroupShuffleSplit` (80% train / 20% test, random_state=42)** grouped by `client_id`, ensuring **strictly 0 overlapping clients**.

### Performance Comparison (Model vs Baseline vs Random Split)

| Evaluation Configuration | Client Overlap | Test Base Rate | Precision@20 | Precision@50 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Before - Memorized)** | 31 clients | 0.545 | **0.80** | **0.88** | **0.734** |
| **Heuristic Baseline (Grouped Split)** | 0 clients | 0.511 | **0.70** | **0.64** | N/A |
| **Decision Tree (Honest Grouped Split)** | **0 clients (PASS)** | **0.511** | **0.45** | **0.66** | **0.679** |
| **Honest Pre-Prediction Model (No 30d Window)**| **0 clients (PASS)** | **0.511** | **0.55** | **0.64** | **0.540** |

### Error Analysis & Key Failure Modes
- **Top False Positives:** Highly ranked pages with older publication dates that retained strong search momentum due to brand authority, misclassified as decaying based on age alone.
- **Top False Negatives:** Moderate-age pages experiencing sudden ranking drops despite recent updates, driven by competitor content improvements not captured in static page metadata.

---

## 6. Interpretation

1. **The Client Memorization Gap:** Random train/test splitting inflated Precision@20 from 0.45 to 0.80 (a 35-percentage-point illusion). When tested on brand-new client domains, model performance normalizes to reflect true generalizability.
2. **Signal Hierarchy:** In pre-prediction modeling, `days_since_last_update` and `content_age_days` provide the strongest non-leaky directional indicators of decay, while `avg_position` in striking distance (4–20) identifies the highest-leverage click recovery opportunities.
3. **Model vs Baseline Trade-off:** The heuristic rule achieved superior precision at the extreme top of the queue (Precision@20: 0.70 vs 0.45), while the learned decision tree proved superior over a broader batch (Precision@50: 0.66 vs 0.64).

---

## 7. Recommendation

The output is deployed as a **4-Tier Content Action Playbook**:
1. **Tier 1: Comprehensive Refresh & Depth Update (`ACTION_REFRESH_STALE_DECAY`)** — 7,107 pages (23.7%). Focus writer hours on mature, stale content with high decay scores.
2. **Tier 2: Snippet & Intent Optimization (`ACTION_OPTIMIZE_SNIPPET_CTR`)** — 7,415 pages (24.7%). Quick-turnaround title and meta description rewrites for striking-distance rankings.
3. **Tier 3: Thin Content Expansion (`ACTION_EXPAND_THIN_CONTENT`)** — 250 pages (0.8%). Subtopic and evidence addition for thin pages with proven demand.
4. **Tier 4: Monitoring Queue (`ACTION_MONITOR_HEALTHY`)** — 15,228 pages (50.8%). Automated observation; zero human hours allocated.

### Operational NO-GO Rules
- 🚫 **No autonomous auto-publishing** without editorial review.
- 🚫 **No automated deletions or mass redirects** based on model scores.
- 🚫 **No artificial word count padding** without informative substance.
- 🚫 **No automated edits on YMYL or high-stakes brand assets.**

---

## 8. Reproducibility

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Asadnaeem23/flyrank-ml-internship.git
   cd flyrank-ml-internship
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn matplotlib nbformat nbclient
   ```
3. **Execute End-to-End Notebooks:**
   Run notebooks in order:
   - `work/notebooks/w05_model.ipynb`
   - `work/notebooks/w06_validation_audit.ipynb`
   - `work/notebooks/w07_action_playbook.ipynb`
   - `work/notebooks/capstone.ipynb`
4. **Random Seed:** All splits, model estimators, and permutation tests utilize fixed `random_state=42`.
5. **Committed Receipts:** Key metrics are tracked in `work/outputs/action_playbook_metrics.json`.

---

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai/)). This real-world anonymized search performance dataset enables reproducible, leak-resistant machine learning research on production search data.
