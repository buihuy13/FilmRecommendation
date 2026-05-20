# Evaluation System Guide

## Overview

This document explains how the Film Recommendation System evaluates recommendation quality. The evaluation consists of **three main assets** that measure different aspects of the system:

1. `evaluate_als` - Evaluates the ALS collaborative filtering model
2. `evaluate_hybrid_sampled` - Evaluates the hybrid recommendation system
3. `evaluate_candidate_coverage` - Diagnostics to identify where items are lost

---

## 1. ALS Evaluation (`evaluate_als`)

### Purpose
Measures the **prediction accuracy** of the ALS (Alternating Least Squares) collaborative filtering model.

### How It Works

#### Step 1: Chronological Train/Test Split
```
For each user:
1. Order all interactions by timestamp
2. Split at 80% chronological cutoff
   - Training: First 80% of interactions (or earliest if user has < 5 ratings)
   - Testing: Last 20% of interactions
```

#### Step 2: User Mean Normalization
```
ALS predicts normalized ratings:
prediction_raw = prediction + user_mean
```
This centers each user's ratings around their personal average, helping the model focus on **relative preferences** rather than absolute values.

#### Step 3: Sampling & Prediction
```python
# Sample 20% of test data for faster evaluation
sample = test_df.sample(False, 0.2, seed=42)

# Get predictions from trained ALS model
predictions = model.transform(sample)
```

#### Step 4: Metrics Calculation

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **RMSE** | √(Σ(pred - actual)² / n) | Average error magnitude (lower is better) |
| **MAE** | Σ\|pred - actual\| / n | Average absolute error (lower is better) |
| **Prediction Coverage** | predictions_count / test_count | % of test items that got a prediction |

**Why RMSE > MAE?** RMSE penalizes large errors more heavily than MAE due to squaring.

### Interpretation

| RMSE Value | Interpretation |
|------------|----------------|
| < 0.8 | Excellent - Predictions are very close to actual ratings |
| 0.8 - 1.0 | Good - Reasonable prediction accuracy |
| 1.0 - 1.2 | Fair - Some deviation from actual ratings |
| > 1.2 | Poor - Predictions are not very accurate |

**Note**: For recommendation systems, RMSE is less important than ranking quality. A model can have high RMSE but still recommend relevant items well.

---

## 2. Hybrid Evaluation (`evaluate_hybrid_sampled`)

### Purpose
Measures **retrieval quality** - how well the hybrid system surfaces relevant items in the top-K recommendations.

### How It Works

#### Step 1: Define Test Items
```python
# Get users who have hybrid recommendations
hybrid_users = hybrid_df.select("userId").distinct()

# Get test items from the last 20% of each user's chronological history
# Only include items the user rated HIGHLY (>= 3.0 stars)
test_items = (
    chronological_split(ratings_df)
    .filter(col("rating") >= 3.0)  # Only "liked" items
    .filter(col("row_num") > eval_cutoff)  # Test set only
)
```

**Key Point**: We only test with items users **actually liked** (rating ≥ 3.0). This tests "can we recommend movies the user will enjoy?"

#### Step 2: Sample Test Items
```python
# Take top 3 highest-rated test items per user
test_items_per_user = (
    test_items
    .orderBy(F.desc("rating"), F.desc("timestamp"))
    .limit(3)  # 3 items per user
)
```

#### Step 3: Build Candidate Pool (1 positive + 99 negatives)
```python
for each user:
    for each test_item (positive):
        add test_item to pool with label=1
    
    # Random negatives NOT in user's history
    neg_pool = all_movies - user_seen - test_items
    negatives = random.sample(neg_pool, 99)
    for neg in negatives:
        add neg to pool with label=0
```

**Why 99 negatives?** This creates a realistic scenario: the system must distinguish 1 relevant item from 99 irrelevant ones.

#### Step 4: Score & Rank
```python
# Join with hybrid recommendations to get scores
scored = candidate_pool.join(hybrid_df, ["userId", "movieId"], "left")

# Rank by hybrid_score (descending)
ranked = scored.orderBy(F.desc("hybrid_score"), F.desc("movieId"))
```

#### Step 5: Calculate Metrics

**Hit@K** - Did the test item appear in the top K positions?
```python
hit_at_k = (rank_in_pool <= K) ? 1 : 0
avg_hit_at_k = mean(hit_at_k for all test items)
```

| Metric | K Value | Interpretation |
|--------|---------|----------------|
| Hit@1 | Top 1 | % of times the #1 recommendation was a test item |
| Hit@5 | Top 5 | % of times test item appeared in top 5 |
| Hit@10 | Top 10 | % of times test item appeared in top 10 |
| Hit@20 | Top 20 | % of times test item appeared in top 20 |

**NDCG@K** - Normalized Discounted Cumulative Gain
```python
ndcg_at_k = sum(1 / log2(rank + 1)) for each hit at rank ≤ K
```

NDCG rewards higher ranks more (hit at rank 1 contributes more than hit at rank 5).

### Interpretation

| Hit@20 Value | Interpretation |
|--------------|----------------|
| > 0.30 | Excellent - 30% of test items in top 20 |
| 0.20 - 0.30 | Good - 20-30% retrieval rate |
| 0.10 - 0.20 | Fair - Some relevant items surface |
| < 0.10 | Poor - Most relevant items missed |

**Why are values typically low?**
- Test items are from the user's future interactions (unseen during training)
- Only 20 slots per user, but 3 test items per user
- Maximum theoretical Hit@20 = 20/(20+3) ≈ 87% if perfect, but realistic systems are much lower

---

## 3. Candidate Coverage Evaluation (`evaluate_candidate_coverage`)

### Purpose
**Diagnostic tool** to identify WHERE in the pipeline test items are being lost.

### How It Works

Measures coverage at 4 stages of the recommendation pipeline:

```
Stage 1: Any Hybrid Coverage
    └── Is the test item ANYWHERE in hybrid recommendations?
        
Stage 2: Collaborative Coverage
    └── Does test item have a collaborative score (> 0)?
        
Stage 3: Content Coverage
    └── Does test item have a content score (> 0)?
        
Stage 4: Top-K Coverage
    └── Is test item in the final top-K recommendations?
```

### Metrics

| Metric | What It Tells You |
|--------|-------------------|
| `any_hybrid_coverage` | % of test items that made it into hybrid pool at all |
| `collab_only_coverage` | % surfaced by collaborative filtering |
| `content_only_coverage` | % surfaced by content-based filtering |
| `top_k_coverage` | % that survived ranking to make top-K |
| `avg_rank_in_top_k` | Average position of test items that made top-K |

### Diagnostic Use Cases

| Symptom | Diagnosis |
|---------|-----------|
| Low `collab_coverage` | Collaborative filtering missing relevant items |
| Low `content_coverage` | Content similarity not finding similar movies |
| High `any_coverage`, low `top_k_coverage` | Items found but not ranked high enough |
| Low `any_coverage` overall | Test items not in candidate pool at all |

---

## Evaluation Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SPLIT                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  Train Set   │         │   Test Set   │                      │
│  │ (80% chrono) │         │ (20% chrono) │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  Train ALS   │         │ Test Items   │                      │
│  │    Model     │         │ (rating ≥ 3) │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         ▼                        │                               │
│  ┌──────────────┐                │                               │
│  │ Generate     │                │                               │
│  │ Hybrid Recs  │                │                               │
│  └──────┬───────┘                │                               │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌──────────────────────────────────────────┐                   │
│  │         EVALUATION                        │                   │
│  │  ┌────────────────┐  ┌─────────────────┐ │                   │
│  │  │  evaluate_als  │  │ evaluate_hybrid │ │                   │
│  │  │  RMSE, MAE     │  │  Hit@K, NDCG@K  │ │                   │
│  │  └────────────────┘  └─────────────────┘ │                   │
│  │                                           │                   │
│  │  ┌───────────────────────────────────┐   │                   │
│  │  │ evaluate_candidate_coverage        │   │                   │
│  │  │ (diagnostics)                      │   │                   │
│  │  └───────────────────────────────────┘   │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Configuration Values

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `HIGH_RATING_THRESHOLD` | 3.0 | `gold.py` | Only ratings ≥ 3.0 are "positive" for testing |
| `EVAL_SAMPLE_FRACTION` | 0.2 | `evaluate.py` | 20% of test data sampled for speed |
| `HYBRID_TOP_K` | 20 | `gold.py` | Number of recommendations per user |
| `NUM_NEGATIVES` | 99 | `evaluate.py` | Negative samples per test item |
| `RECS_USER_SAMPLE` | 2000 | `evaluate.py` | Users sampled for ALS evaluation |

---

## Improvements Made (2025-05-19)

The following improvements were implemented to increase evaluation reliability:

1. **Increased Candidate Pool**: More candidates per user → better chance of including test items
2. **Removed 0.2 Threshold**: Stopped filtering valid collaborative candidates
3. **Fixed Ranking**: Ensured recommendations are properly sorted before evaluation
4. **Added Coverage Diagnostic**: New asset to identify where items are lost

### Expected Impact

| Change | Expected Metric Improvement |
|--------|----------------------------|
| Remove 0.2 threshold | +15-25% Hit@K |
| Increase collab candidates (1000→2000) | +10-20% Hit@K |
| Increase content seeds (10→20) | +5-15% Hit@K |
| Fix ranking logic | More consistent results |

---

## Running Evaluations

### Via Dagster UI
1. Navigate to the "Assets" tab
2. Select evaluation assets:
   - `evaluate_als`
   - `evaluate_hybrid_sampled`
   - `evaluate_candidate_coverage`
3. Click "Materialize"

### Viewing Results
Each evaluation asset outputs metadata:
- RMSE, MAE, coverage for ALS
- Hit@1, Hit@5, Hit@10, Hit@20, NDCG@K for hybrid
- Coverage percentages at each pipeline stage

---

## FAQ

**Q: Why is Hit@K often low even with good recommendations?**
A: The evaluation tests if the system can predict SPECIFIC items the user rated highly. This is harder than general recommendation quality.

**Q: Should I prioritize RMSE or Hit@K?**
A: For recommender systems, Hit@K is more important. Users care about relevant items in top recommendations, not precise rating predictions.

**Q: What if candidate coverage is low?**
A: Use `evaluate_candidate_coverage` to diagnose. If collab coverage is low, ALS may need tuning. If content coverage is low, content features may need improvement.

**Q: Can I change the number of test items per user?**
A: Yes, modify the `filter(col("rn") <= 3)` line in `evaluate_hybrid_sampled`. Fewer test items = higher Hit@K but less robust evaluation.

---

## References

- **Gold Layer Implementation**: `pipeline/assets/gold.py`
- **Evaluation Implementation**: `pipeline/assets/evaluate.py`
- **Improvement Recommendations**: `docs/IMPROVEMENT_RECOMMENDATIONS.md`
