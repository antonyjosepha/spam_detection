# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

Builds directly on Part 2's exact feature matrix and train/test split (same
`random_state=42`, `test_size=0.2` — `part3_processing.prepare_data()` reproduces the
identical row partition, confirmed by matching Part 2's Logistic Regression numbers when
refit here). **Runtime note:** Task 6's GridSearchCV is 90 RandomForest fits and took
~2 minutes end-to-end over real HTTP in testing — this is inherent to exhaustive grid
search, not an inefficiency in the implementation.

## Task 1–2 — Decision Tree: Unconstrained vs. Controlled

| Tree | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Unconstrained (max_depth=None) | 0.9993 | 0.7529 | **0.2464** |
| Controlled (max_depth=5, min_samples_split=20) | 0.7566 | 0.7511 | **0.0055** |

The unconstrained tree shows clear **overfitting**: 99.93% train accuracy vs. 75.29% test
— a 24.6-point gap. It grew to depth 39 with 5,038 leaves, meaning it kept splitting until
(in many branches) it isolated individual or near-individual training rows. Decision trees
are **high-variance models** because they build greedily: at each node, the algorithm picks
the single best split for *that* node given the data that reached it, and never revisits or
reconsiders earlier splits once made. With no depth or sample constraint, this greedy process
keeps carving up the training set until it fits noise as readily as signal — a small change
in the training data can produce a very different tree.

`max_depth` limits how many splits deep the tree can grow, directly capping model
complexity — this reduces variance (less opportunity to fit noise) at the cost of some bias
(the tree can no longer represent very fine-grained decision boundaries).
`min_samples_split=20` prevents splitting a node with fewer than 20 samples, which stops the
tree from creating splits driven by tiny, noisy subsets that don't generalize. Together they
cut the train/test gap from 0.2464 to 0.0055 — a controlled tree that barely overfits at all,
at a modest cost to peak train accuracy (99.93% → 75.66%).

## Task 3 — Gini vs. Entropy (max_depth=5)

**Gini impurity:** `Gini = 1 - Σ pᵢ²`
**Entropy:** `Entropy = -Σ pᵢ log₂(pᵢ)`

where `pᵢ` is the proportion of class `i` samples at a node. **Gini = 0** means the node is
**pure** — every sample at that node belongs to a single class, so no further split could
reduce impurity there.

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.7511 |
| Entropy | 0.7490 |

Nearly identical (0.21-point difference) — expected, since Gini and Entropy usually select
very similar splits in practice; they diverge only in edge cases with more than two classes
or unusual class-probability distributions, neither of which applies to this binary
spam/ham task.

## Task 4 — Random Forest

| Train Acc | Test Acc | AUC |
|---|---|---|
| 0.8172 | 0.7918 | **0.8728** |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| exclaim_count | 0.3220 |
| avg_word_len | 0.1870 |
| digit_count | 0.1673 |
| punct_count | 0.0969 |
| weekday_Saturday | 0.0754 |

**How Random Forest computes feature importance:** for each feature, it averages the
reduction in Gini impurity achieved by splits on that feature, across every split in every
tree in the forest, weighted by how many samples reach each split. This differs fundamentally
from a linear regression coefficient: a regression coefficient is a single global,
directional, linear weight (holding other features fixed), while Random Forest importance is
a **non-linear, non-directional** measure of how useful a feature was for reducing prediction
error across hundreds of different splitting contexts — it says "how much did this feature
help," not "in which direction and by how much."

**Bagging:** each of the 100 trees is trained on a **bootstrap sample** — a random sample of
the training data drawn *with replacement*, the same size as the original training set, so
some rows appear multiple times and others not at all in a given tree's sample. At each split,
only a **random subset of √(number of features)** ≈ √12 ≈ 3 features are considered, rather
than all 12 — this decorrelates the trees, since without it, a few dominant features
(`exclaim_count`, `avg_word_len`) would drive nearly every tree to make similar early splits.
Averaging predictions across many trees that were each trained on different data and
considered different feature subsets **cancels out** the idiosyncratic overfitting of any
single tree — individual trees' errors are only weakly correlated, so the ensemble's variance
shrinks roughly in proportion to how uncorrelated the trees' errors are, without the deep
single-tree overfitting seen in Task 1.

## Task 4a — Gradient Boosting

| Train Acc | Test Acc | AUC |
|---|---|---|
| 0.7943 | 0.7855 | 0.8655 |

Slightly below Random Forest on all three metrics on this dataset (AUC 0.8655 vs. 0.8728),
though both are much stronger than either single decision tree.

## Task 4b — Feature Ablation

5 lowest-importance features removed: `weekday_Monday, weekday_Thursday, weekday_Wednesday,
weekday_Tuesday, capital_ratio`.

| Model | Test AUC |
|---|---|
| Full (all 12 features) | 0.8728 |
| Reduced (7 features) | **0.8789** |

AUC **improved** (by 0.0060) after removing these 5 features — they were **genuinely
uninformative**, not just weakly informative. This makes sense: `capital_ratio` was already
flagged in Part 1 as a near-degenerate column (only 43/30,494 rows nonzero), and 4 of 5
individual weekday dummies carry little signal once `weekday_Saturday` (the one weekday that
did rank in the top 5) captures most of the day-of-week effect. Removing pure noise features
can help a tree-based model slightly, since it removes candidate splits that occasionally get
selected on spurious in-sample patterns.

**Production implication:** a 7-feature model that matches or slightly beats a 12-feature
model is an easy call — fewer features means lower inference cost, a smaller serialized
model, less data-pipeline maintenance (nothing downstream needs to keep computing
`capital_ratio` or four weekday flags), and one less place for future data drift to cause
problems. This trade would only become risky if AUC had *dropped* meaningfully on removal;
here it didn't, so simplifying is a clear win with no accuracy cost.

## Task 5 — Cross-Validated Comparison (5-fold StratifiedKFold, ROC-AUC)

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.7852 | 0.0071 |
| Decision Tree (max_depth=5) | 0.8244 | 0.0047 |
| Random Forest | 0.8733 | 0.0037 |
| Gradient Boosting | 0.8677 | 0.0034 |

**Why cross-validation beats a single train-test split:** a single split's reported
performance depends partly on which rows happened to land in the test set — an unusually
easy or hard test split can make a model look better or worse than it really is. 5-fold CV
evaluates the model on 5 different train/test partitions and averages the results, which
both reduces the influence of any one lucky/unlucky split and provides a **standard
deviation** (shown above) — a direct measure of how sensitive each model's performance is to
which rows end up in the test fold. Random Forest's low std (0.0037) indicates consistently
strong performance across folds; Logistic Regression's higher std (0.0071) shows it's
somewhat more sensitive to the specific data split.

## Task 6 — GridSearchCV (Random Forest Pipeline)

Grid: `n_estimators` [50,100,200] × `max_depth` [5,10,None] × `min_samples_leaf` [1,5] =
**18 configurations × 5 folds = 90 total model fits**.

**Best params:** `max_depth=None, min_samples_leaf=5, n_estimators=200`
**Best CV AUC:** 0.8833  |  **Test AUC:** 0.8855

**Grid Search vs. Randomized Search trade-off:** Grid Search is exhaustive — it tries every
combination, guaranteeing the best combination *within the specified grid* is found, at a
cost that grows multiplicatively with each added parameter or value (adding one more
`n_estimators` option here would mean +30 fits). Randomized Search instead samples a fixed
number of random combinations from the parameter space, which scales independently of grid
size — it can cover a much larger or continuous parameter space in the same budget, at the
cost of no longer guaranteeing the single best combination is found. With only 18
combinations here, exhaustive search was cheap enough to be worth the guarantee; a larger
grid (e.g. adding `min_samples_split`, more `n_estimators` values) would tip the balance
toward Randomized Search.

## Task 6b — Learning Curve (Tuned Pipeline)

| Training Fraction | N Rows | Train AUC | Test AUC |
|---|---|---|---|
| 0.2 | 4,879 | 0.9481 | 0.8651 |
| 0.4 | 9,758 | 0.9475 | 0.8760 |
| 0.6 | 14,637 | 0.9505 | 0.8786 |
| 0.8 | 19,516 | 0.9500 | 0.8828 |
| 1.0 | 24,395 | 0.9505 | 0.8855 |

**(i) Does training AUC decrease as the training set grows?** No — it stays essentially flat
(~0.948–0.951) across all five fractions. This is *not* the classic shrinking-train-score
pattern of a high-variance model overfitting small datasets; even at 20% of the data (4,879
rows), the tuned Random Forest already fits the training set about as tightly as it ever
will.

**(ii) Does test AUC increase with more data?** Yes, steadily: 0.8651 → 0.8855 across the
five fractions, with no sign of flattening out even at 100% of available training data.

**(iii) Conclusion — data-limited or capacity-limited?** **Data-limited.** Test AUC is still
climbing at the full training set size with no plateau, while train AUC has already been flat
since the smallest fraction tested — meaning the model isn't struggling to fit the data it
has (that's not the bottleneck), it's struggling to *generalize* from a training set that's
still too small relative to the problem's complexity. The persistent train/test AUC gap
(~0.06–0.09) further confirms there's remaining variance the model hasn't resolved.
**Practical implication: collecting more labeled email data would likely improve this
model's performance further** — increasing model capacity (e.g. deeper trees, more
estimators) is unlikely to help much on its own, since the ceiling here appears to be data
volume, not the model's ability to fit what it's already been given.

## Task 7 — Serialized Model

Saved via `joblib.dump(best_pipeline, 'best_model.pkl')` to the run directory. Reload +
predict smoke test:

```python
import joblib

model = joblib.load('best_model.pkl')
sample_rows = X_test.iloc[:2]   # two hand-crafted / held-out rows, unscaled --
                                 # the pipeline's own imputer+scaler handles preprocessing
predictions = model.predict(sample_rows)
print(predictions)              # -> [0, 0]  (both predicted "ham")
```
Confirmed to run without errors (see server log: `task7: reload+predict on 2 sample rows ->
[0, 0]`).

## Task 8 — Summary Comparison (Parts 2 + 3)

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression (Part 2) | 0.7852 | 0.0071 | 0.7858 |
| Decision Tree (max_depth=5) | 0.8244 | 0.0047 | 0.8216 |
| Random Forest | 0.8733 | 0.0037 | 0.8728 |
| Gradient Boosting | 0.8677 | 0.0034 | 0.8655 |
| **Tuned RF Pipeline (GridSearchCV)** | **0.8833** | — | **0.8855** |

**Recommendation: the Tuned Random Forest Pipeline (`max_depth=None, min_samples_leaf=5,
n_estimators=200`).** It has the highest CV mean AUC and highest test AUC of every model
built across Parts 2–3, with test AUC (0.8855) actually slightly *exceeding* its CV mean
(0.8833) rather than falling short of it — a good sign against overfitting to the tuning
process. It's also the only model here packaged as a complete, reproducible `Pipeline`
(imputation + scaling + model in one object), meaning it can be deployed directly on raw
`X_test`-shaped input without a separate manual preprocessing step, unlike the other models
which require the caller to replicate Part 2's scaling externally. The one caveat, per Task
6b: this model is currently data-limited, so its performance should be expected to keep
improving if more labeled email data becomes available — it is not yet at a capacity
ceiling.

## Acceptance criteria checklist
- [x] Runs top-to-bottom without errors (confirmed via real HTTP request, ~2.5 min).
- [x] Constrained vs. unconstrained decision tree train/test gap comparison — in README.
- [x] Gini and Entropy formulas written out — above.
- [x] Top 5 feature importances reported and explained.
- [x] GridSearchCV best_params_ + best_score_ printed/logged and in README.
- [x] `best_model.pkl` produced per run (see Task 7; regenerated by `run_part3_pipeline()`).
- [x] Reload-and-predict code block runs without errors.
- [x] Summary comparison table with final recommendation and justification — above.
- [x] GradientBoostingClassifier trained and included in the Task 5 CV comparison.
- [x] 5 lowest-importance features identified, second Random Forest trained without them,
      both AUCs reported and compared, uninformative-vs-contributing interpreted, production
      trade-off discussed.
- [x] Pipeline trained at 5 fractions (20–100%), train/test AUC printed for each, README
      interprets the trend and concludes data-limited vs. capacity-limited.
