# ADVANCED ML TOPICS: ASSIGNMENTS 7-14 SUMMARY

## Complete Guide to Advanced Classification Evaluation & Optimization

Successfully implemented all advanced machine learning topics with practical examples on the fraud detection dataset.

---

## ASSIGNMENT 7: Precision and Recall

### What They Measure

**Precision**: "Of all cases the model predicted as positive, how many were actually positive?"
```
Precision = TP / (TP + FP)
Measures: Trustworthiness of positive predictions
```

**Recall**: "Of all actual positive cases, how many did the model correctly detect?"
```
Recall = TP / (TP + FN)
Measures: Completeness of positive detection
```

### Why They Matter
- **Accuracy alone is misleading** on imbalanced data
- Example: On 90% legitimate dataset, predicting "all legitimate" = 90% accuracy but 0% fraud recall (useless!)
- Precision and Recall expose what accuracy hides

### When Each Matters
- **Precision Priority** (False Positives costly):
  - Spam filtering: blocking legitimate emails is bad
  - Content moderation: removing correct posts harms users
  - Loan approval: falsely approving bad loans costs money

- **Recall Priority** (False Negatives costly):
  - Fraud detection: missing fraud = direct financial loss
  - Disease screening: missing illness delays treatment
  - Security: missing intrusion causes catastrophic damage

### Key Insight
There is a **fundamental trade-off**:
- Lower decision threshold → Higher Recall, Lower Precision
- Higher decision threshold → Higher Precision, Lower Recall
- Different problems demand different thresholds

---

## ASSIGNMENT 8: F1-Score

### Definition
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The **harmonic mean** of Precision and Recall — heavily penalizes extreme imbalance.

### Why Harmonic Mean?
| Metric | P=0.9, R=0.1 | Meaning |
|--------|--------------|---------|
| Arithmetic Mean | 0.50 | Misleading — suggests mediocre |
| Harmonic Mean (F1) | 0.18 | Honest — reveals near-failure |

**F1 prevents gaming**: A model can't get high F1 by excelling at one metric while failing at the other.

### When to Use F1
- **Best for**: Imbalanced classification where both error types matter
- **Default metric** for model selection on imbalanced data
- **Cannot ignore either** Precision or Recall

### When F1 Is Wrong
- One error type far more costly → Use Fβ-score to weight accordingly
- True Negatives matter → Use Balanced Accuracy or MCC
- Probability calibration matters → Use ROC-AUC or Brier Score

---

## ASSIGNMENT 9: Confusion Matrix

### Structure (Binary Classification)
```
                Predicted Positive    Predicted Negative
Actual Positive      TP (correct)         FN (missed)
Actual Negative      FP (false alarm)     TN (correct)
```

### Cell Meanings

| Cell | Name | Cost | Example |
|------|------|------|---------|
| TP | True Positive | ✓ Correct | Fraud caught |
| TN | True Negative | ✓ Correct | Legitimate cleared |
| FP | False Positive | ✗ False alarm | Legitimate blocked |
| FN | False Negative | ✗ Missed | Fraud slipped through |

### Why Inspect It
- **Reveals asymmetry**: FP and FN costs are rarely equal
- **Exposes what metrics hide**: Accuracy can be 95% while catching 0% of the positive class
- **Guides threshold tuning**: Adjusting threshold reshapes every cell
- **Enables domain validation**: Non-experts can judge if the model "makes sense"

### Key Principle
Every metric (Precision, Recall, F1) derives from these four cells. The confusion matrix is the source of truth.

---

## ASSIGNMENT 10: K-Nearest Neighbors

### Core Algorithm
1. Choose K (number of neighbors)
2. Find K closest training examples to new sample
3. Majority vote (classification) or average (regression)

### No Learning Phase
- Unlike Linear Regression or Decision Trees
- KNN is "lazy" — defers work to prediction time
- Entire training set is the model

### Key Requirements
1. **Feature Scaling (NON-NEGOTIABLE)**
   - Without scaling, large-scale features dominate distance
   - Example: Income (range 20K-2M) overwhelms Age (18-60)
   - Always use: `StandardScaler` in pipeline

2. **Choosing K** — Critical Hyperparameter
   - K=1: High variance, memorizes noise
   - K=50: High bias, boundary too smooth
   - Optimal K via cross-validation, typically √n

### Bias-Variance on Display
| K Value | Bias | Variance | Typical |
|---------|------|----------|---------|
| 1 | Low | High | Overfitting |
| Medium (5-10) | Balanced | Balanced | Often optimal |
| 50+ | High | Low | Underfitting |

### Strengths
- No assumptions about data distribution
- Naturally handles non-linearity
- Interpretable: see nearest neighbors driving prediction
- Works on small datasets

### Weaknesses
- **Curse of Dimensionality**: degrades with >20 features
- **Slow at prediction**: O(n×d) for each inference
- **Memory intensive**: must store entire training set
- Sensitive to irrelevant features (adds noise to distances)

---

## ASSIGNMENT 11: Bias-Variance Trade-Off

### Fundamental Concept
```
Total Error = Bias² + Variance + Irreducible Noise
```

**Irreducible Noise** is the hard floor — no algorithm can improve beyond it.

### Bias: Error from Wrong Assumptions
- **High Bias** = Underfitting
- Model is too simple, can't capture real patterns
- Signature: Both train AND test error are high, similar to each other
- Example: Linear model on quadratic data

### Variance: Error from Training Data Sensitivity
- **High Variance** = Overfitting
- Model is too complex, memorizes training-specific noise
- Signature: Train error very low, test error high, large gap
- Example: KNN with K=1 on noisy data

### The U-Curve

```
Error
  │     High Bias       High Variance
  │         ↓                ↓
  │         \                /
  │          \              /
  │           \            /
  │            \          /
  │            Optimal →  /
  │            \_       _/
  └────────────────────────────
    Low Complexity          High Complexity
```

As complexity increases:
- Bias ↓ (model can capture more patterns)
- Variance ↑ (model more sensitive to training data)
- There is NO setting that eliminates both

### Diagnostics

| Train Performance | Test Performance | Train/Test Gap | Diagnosis |
|------------------|-----------------|----------------|---------  |
| Poor | Poor | Small | High Bias |
| Excellent | Poor | Large | High Variance |
| Good | Good | Small | Good Fit |
| Poor | Very Poor | Large | Both (data issue) |

### Solutions

**For High Bias** (underfitting):
- Increase model complexity
- Add more features / feature engineering
- Reduce regularization
- Deeper trees, smaller K, lower degree regularization

**For High Variance** (overfitting):
- Collect more training data (most robust)
- Apply regularization (L1, L2, dropout)
- Reduce model complexity
- Increase K, reduce tree depth, higher regularization strength

### Key Principle
You CANNOT eliminate both simultaneously. The trade-off is mathematical, not a modeling failure. Your job is to find the optimal balance for your specific problem.

---

## ASSIGNMENT 12: Decision Trees

### How They Work
1. Start with all training data at root
2. Find best split: (feature, threshold) that maximizes impurity reduction
3. Create two child nodes: left (true) and right (false)
4. Recursively repeat until stopping criteria met

### What "Best Split" Means
- **Classification**: Minimize Gini impurity or Entropy
- **Regression**: Minimize MSE (variance) of target values
- Weighted by number of samples at each node

### Stopping Criteria
```python
max_depth          # Maximum tree depth
min_samples_split  # Min samples to split a node
min_samples_leaf   # Min samples required in leaf
```

**Without constraints** → Tree memorizes every training point → 100% train accuracy, poor test accuracy

### Key Properties
1. **No feature scaling needed** — splits based on ranked order
2. **Automatic feature interactions** — splits at depth N depend on earlier splits
3. **Interpretable** — every prediction reads as a flowchart of decisions
4. **High variance** — small data changes can produce different trees

### Visualization & Rules
```python
from sklearn.tree import plot_tree

plot_tree(tree, feature_names=columns, class_names=['Legit', 'Fraud'])
```

The tree can be printed and inspected — make sure the rules make sense!

### Feature Importance in Trees
```
Importance = Sum of weighted impurity reductions
```

Features that split earlier and affect more samples accumulate higher importance.

---

## ASSIGNMENT 13: Feature Importance

### Two Methods

**1. Mean Decrease in Impurity (MDI)**
- How much did each feature reduce impurity during splits?
- Built-in: `model.feature_importances_`
- Fast (computed during training)
- Biased toward high-cardinality features

**2. Permutation Importance**
- Shuffle each feature, measure performance drop
- More reliable: does the model actually rely on this feature?
- Works with any model (not just trees)
- More computationally expensive

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
```

### Key Warnings

**Importance ≠ Causation**
- High importance = predictively useful in this model
- Does NOT mean the feature causes the outcome
- Example: Customer tenure predicts churn, but may correlate with contract type

**Conditional on Other Features**
- If two features are correlated, model uses whichever it encounters first
- One gets high importance; other gets zero
- They carry the same signal, but only one is credited

**Dataset & Model Specific**
- Train on different data → different importance ranking
- Retrain with different random seed → potentially different ranking
- For unstable features, validate with permutation importance

### Practical Use
1. **Model debugging**: Why is X feature important when it shouldn't be?
2. **Feature selection**: Remove low-importance features
3. **Business insight**: What drives fraud/churn?
4. **Leakage detection**: Suspicious high importance on "non-predictive" features?

### Best Practice
- Report MDI for exploration
- Validate with permutation importance for decisions
- Check correlation matrix for feature groups
- Never remove a feature without retraining to verify impact

---

## ASSIGNMENT 14: GridSearchCV Hyperparameter Tuning

### What Are Hyperparameters?
Settings defined **before training** that control model behavior:
- Not learned from data
- Set by practitioner
- **Cannot be determined from first principles** — must be searched

### Hyperparameter Impact
| Model | Hyperparameter | High Value | Low Value |
|-------|----------------|-----------|-----------|
| KNN | K | High Bias | High Variance |
| Tree | max_depth | High Variance | High Bias |
| LogReg | C | High Variance | High Bias |
| SVM | C | High Variance | High Bias |

### How GridSearchCV Works
```
1. Define grid: {'param1': [v1, v2, v3], 'param2': [v4, v5]}
2. Generate combinations: All Cartesian products
3. Cross-validate each: 5-fold CV on training set
4. Select best: Highest mean CV score
5. Refit on full training data: Final model uses all training examples
```

### Critical: Use Pipelines
```python
# CORRECT — prevents leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])
grid = GridSearchCV(pipeline, param_grid, cv=5)

# WRONG — scaling leaks validation fold info
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← Leakage!
grid = GridSearchCV(model, param_grid, cv=5)
```

Why: Scaler fitted on full data sees validation fold samples → inflates CV scores.

### Metric Choice Matters
```python
# Imbalanced data — use these
GridSearchCV(..., scoring='f1')
GridSearchCV(..., scoring='roc_auc')

# Balanced data or if accuracy appropriate
GridSearchCV(..., scoring='accuracy')

# Regression
GridSearchCV(..., scoring='neg_mean_squared_error')
```

### Example: Decision Tree
```python
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}
# 5 × 4 × 2 = 40 combinations × 5 folds = 200 model fits
```

### Interpreting Results
```python
results_df = pd.DataFrame(grid.cv_results_)

# Top configurations by score
results_df.nsmallest(5, 'rank_test_score')[[
    'params', 'mean_test_score', 'std_test_score'
]]
```

What to look for:
- Is `std_test_score` low? (stable across folds)
- Are multiple configs scoring similarly? (choose simpler one)
- Is best score at edge of grid? (extend grid)

### Computational Costs
```
Number of fits = n_combinations × n_folds

Example: 4 hyperparams × 5 values each × 5 folds = 2,500 fits
For slow models (Random Forest), this becomes prohibitive
```

**Solutions**:
- Use `n_jobs=-1` for parallelization
- Use `RandomizedSearchCV` for large grids
- Tune sequentially: fix best param for next tune
- Use coarse-then-fine strategy: 2 iterations with widening

### RandomizedSearchCV Alternative
For large search spaces, randomly sample combinations:
```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=50,  # Sample 50 random combinations
    cv=5,
    random_state=42
)
```

Often finds solutions within 1-2% of exhaustive search at 10× lower cost.

### CRITICAL: Never Tune on Test Set
```python
# CORRECT — test set untouched during tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
grid.fit(X_train, y_train)  # Tuning here
final_score = grid.best_estimator_.predict(X_test)  # Evaluation here

# WRONG — test set influences tuning
grid.fit(X_test, y_test)  # ← Leakage!
```

Why: If you tune on test data, reported performance is optimistic and won't generalize.

### Transparent Reporting
```
Dataset:         Fraud, 1000 samples
Model:           Logistic Regression with StandardScaler
Grid:            n_neighbors: [1-20], weights: [uniform, distance]
CV:              5-fold stratified
Scoring:         F1

Baseline F1:     0.000 (majority predictor)
Default Model:   0.64  (arbitrary K=5, uniform)
Tuned Model:     0.71 ± 0.03 (K=11, distance)
Test Set:        0.69  (evaluated once after tuning)

Best Parameters: {'n_neighbors': 11, 'weights': 'distance'}
```

Always include:
- Baseline for context
- CV mean AND std (shows stability)
- Final test score
- Best hyperparameters (reproducibility)

---

## Comprehensive Workflow Summary

### Complete Model Development Pipeline

```
1. LOAD & EXPLORE DATA
   ↓
2. SPLIT: Train/Test (NEVER TOUCH TEST UNTIL END)
   ↓
3. BASELINE: Establish performance floor
   ↓
4. SIMPLE MODEL: Linear/Logistic regression
   ↓
5. GRIDSEARCHCV: Tune hyperparameters (train only)
   ├── Use appropriate scoring metric
   ├── Use pipeline (prevent leakage)
   ├── Use cross-validation
   └── Inspect CV std and top configurations
   ↓
6. COMPLEX MODELS: Trees, KNN, Ensembles
   ├── Apply same GridSearchCV procedure
   └── Compare multiple model types
   ↓
7. ANALYZE
   ├── Confusion matrix
   ├── Precision/Recall/F1
   ├── Feature importance
   └── Bias-variance diagnostics
   ↓
8. FINAL EVALUATION
   └── Best model evaluated on test set EXACTLY ONCE
```

### Metric Selection by Problem

| Problem | Class Balance | Primary Metric | Secondary |
|---------|---------------|----------------|-----------|
| Fraud Detection | Severe (1-10%) | Recall | F1, ROC-AUC |
| Spam Filtering | Severe (1-5%) | Precision | F1 |
| Disease Screening | Moderate (5-20%) | Recall | Precision |
| Churn Prediction | Moderate (10-30%) | F1 | ROC-AUC |
| Ad Click | Severe (0.1-1%) | ROC-AUC | F1 |

---

## Key Takeaways

1. **Precision and Recall are not independent** — there's a fundamental trade-off
2. **F1-Score** balances both, especially useful for imbalanced data
3. **Confusion Matrix** reveals what metrics hide — always inspect it
4. **KNN requires scaling** and K must be cross-validated
5. **Bias-Variance is unavoidable** — find the optimal balance, don't try to eliminate both
6. **Decision Trees are interpretable** but prone to overfitting without constraints
7. **Feature Importance** is predictive association, not causation
8. **GridSearchCV** is the systematic way to find optimal hyperparameters
9. **Never tune on test data** — this is the most common serious mistake
10. **Use pipelines** to prevent data leakage during CV

---

## Files Generated

- `run_all_assignments.py` - Assignments 1-6 (Baseline through Classification)
- `run_advanced_assignments.py` - Assignments 7-14 (Advanced topics)
- `assignments_complete.ipynb` - Jupyter notebook with all code
- `ASSIGNMENTS_SUMMARY.md` - Complete documentation

---

**Status**: ✅ All 14 Assignments Successfully Completed
**Last Updated**: May 17, 2026
