#!/usr/bin/env python3
"""
Advanced ML Topics: Assignments 7-14
Precision & Recall → F1-Score → Confusion Matrix → KNN → Bias-Variance → 
Decision Trees → Feature Importance → GridSearchCV
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    learning_curve, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, accuracy_score, roc_auc_score, 
    balanced_accuracy_score, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("\n" + "="*80)
print("ADVANCED ML TOPICS: ASSIGNMENTS 7-14")
print("="*80)

# Load and prepare data
df = pd.read_csv('data/raw/fraud_data.csv')
df_encoded = df.copy()

category_encoder = LabelEncoder()
location_encoder = LabelEncoder()
df_encoded['category_encoded'] = category_encoder.fit_transform(df_encoded['category'])
df_encoded['location_encoded'] = location_encoder.fit_transform(df_encoded['location'])

feature_columns = ['amount', 'transaction_count', 'velocity', 'category_encoded', 'location_encoded']
X = df_encoded[feature_columns]
y = df_encoded['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Fraud rate: {y.mean():.1%}")

# ============================================================================
# ASSIGNMENT 7: PRECISION AND RECALL
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 7: PRECISION AND RECALL")
print("="*80)

# Train baseline and logistic regression
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]

# Compute metrics
baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
baseline_rec = recall_score(y_test, baseline_pred, zero_division=0)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec = recall_score(y_test, lr_pred, zero_division=0)

print(f"\nBASELINE (Majority Class):")
print(f"  Precision: {baseline_prec:.1%} (of flagged fraud, how many are real?)")
print(f"  Recall:    {baseline_rec:.1%} (of all fraud, how many caught?)")

print(f"\nLOGISTIC REGRESSION:")
print(f"  Precision: {lr_prec:.1%}")
print(f"  Recall:    {lr_rec:.1%}")

print(f"\n📊 INTERPRETATION:")
print(f"  • Baseline catches {baseline_rec:.1%} of fraud (very bad)")
print(f"  • Model catches {lr_rec:.1%} of fraud (much better)")
print(f"  • Model has {lr_prec:.1%} precision (when it flags fraud, is it right?)")

# Threshold analysis
print(f"\n--- THRESHOLD SENSITIVITY ---")
print(f"Adjusting decision threshold to prioritize recall:\n")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<8}")
print("-" * 45)

for threshold in [0.3, 0.5, 0.7]:
    y_custom = (lr_prob >= threshold).astype(int)
    prec = precision_score(y_test, y_custom, zero_division=0)
    rec = recall_score(y_test, y_custom, zero_division=0)
    f1 = f1_score(y_test, y_custom, zero_division=0)
    print(f"{threshold:<12.1f} {prec:<12.1%} {rec:<12.1%} {f1:<8.3f}")

print(f"\n💡 KEY INSIGHT: Lowering threshold increases recall (catch more fraud)")
print(f"   but decreases precision (more false alarms)")

# ============================================================================
# ASSIGNMENT 8: F1-SCORE
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 8: F1-SCORE (Balancing Precision and Recall)")
print("="*80)

baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

print(f"\nF1-Score = 2 × (Precision × Recall) / (Precision + Recall)")
print(f"Harmonic mean — penalizes extreme imbalance between P and R\n")

print(f"BASELINE F1:        {baseline_f1:.3f}")
print(f"LOGISTIC REG F1:    {lr_f1:.3f}")
print(f"Improvement:       +{lr_f1 - baseline_f1:.3f}")

# Show F1 at different thresholds
print(f"\n--- F1 at Different Thresholds ---")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<8}")
print("-" * 45)

f1_scores = []
thresholds_list = np.arange(0.1, 0.9, 0.1)
for threshold in thresholds_list:
    y_custom = (lr_prob >= threshold).astype(int)
    prec = precision_score(y_test, y_custom, zero_division=0)
    rec = recall_score(y_test, y_custom, zero_division=0)
    f1 = f1_score(y_test, y_custom, zero_division=0)
    f1_scores.append(f1)
    print(f"{threshold:<12.1f} {prec:<12.1%} {rec:<12.1%} {f1:<8.3f}")

best_f1_threshold = thresholds_list[np.argmax(f1_scores)]
print(f"\n✓ Best F1 at threshold: {best_f1_threshold:.1f}")

# ============================================================================
# ASSIGNMENT 9: CONFUSION MATRIX
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 9: CONFUSION MATRIX")
print("="*80)

cm_baseline = confusion_matrix(y_test, baseline_pred)
cm_lr = confusion_matrix(y_test, lr_pred)

print(f"\nBASELINE CONFUSION MATRIX (Majority Class):")
print(f"  Predicted:    Legit  Fraud")
print(f"  Actual Legit:  {cm_baseline[0,0]:>5}  {cm_baseline[0,1]:>5}")
print(f"  Actual Fraud:  {cm_baseline[1,0]:>5}  {cm_baseline[1,1]:>5}")

print(f"\nLOGISTIC REGRESSION CONFUSION MATRIX:")
print(f"  Predicted:    Legit  Fraud")
print(f"  Actual Legit:  {cm_lr[0,0]:>5}  {cm_lr[0,1]:>5}")
print(f"  Actual Fraud:  {cm_lr[1,0]:>5}  {cm_lr[1,1]:>5}")

tn, fp, fn, tp = cm_lr.ravel()
print(f"\nCELL BREAKDOWN:")
print(f"  TP (caught fraud):        {tp}")
print(f"  FN (missed fraud):        {fn}")
print(f"  FP (false alarms):        {fp}")
print(f"  TN (correct legitimate):  {tn}")

print(f"\nFRAUD DETECTION RATE:")
print(f"  Baseline catches: {cm_baseline[1,1]}/{cm_baseline[1,1] + cm_baseline[1,0]} = {cm_baseline[1,1]/(cm_baseline[1,1] + cm_baseline[1,0]):.1%}")
print(f"  Model catches:    {tp}/{tp + fn} = {tp/(tp+fn):.1%}")

# ============================================================================
# ASSIGNMENT 10: K-NEAREST NEIGHBORS
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 10: K-NEAREST NEIGHBORS (KNN)")
print("="*80)

print(f"\nKNN Principle: Similar inputs → Similar outputs")
print(f"Finding K nearest neighbors and letting them vote on prediction\n")

# Test different K values
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

print(f"--- KNN Performance Across Different K Values ---\n")
print(f"{'K':<5} {'Train Acc':<12} {'Test Acc':<12} {'Train/Test Gap':<15} {'Recall':<8}")
print("-" * 55)

for k in [1, 3, 5, 10, 20, 50]:
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_train, y_train)
    
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    gap = train_acc - test_acc
    rec = recall_score(y_test, knn.predict(X_test), zero_division=0)
    
    print(f"{k:<5} {train_acc:<12.3f} {test_acc:<12.3f} {gap:<15.3f} {rec:<8.1%}")

print(f"\n📊 INTERPRETATION:")
print(f"  K=1: High variance (overfits to noise)")
print(f"  K=50: High bias (boundary too smooth)")
print(f"  K=5-10: Often a good balance")

# Train final KNN
knn_final = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_final.fit(X_train, y_train)
knn_pred = knn_final.predict(X_test)

print(f"\nKNN (K=5) Test Accuracy: {accuracy_score(y_test, knn_pred):.3f}")
print(f"KNN (K=5) Test Recall:   {recall_score(y_test, knn_pred, zero_division=0):.1%}")

# ============================================================================
# ASSIGNMENT 11: BIAS-VARIANCE TRADE-OFF
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 11: BIAS-VARIANCE TRADE-OFF")
print("="*80)

print(f"\nBias: Error from oversimplified assumptions (underfitting)")
print(f"Variance: Error from sensitivity to training data (overfitting)\n")

print(f"--- Model Complexity vs Performance ---\n")
print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<8} {'Diagnosis':<20}")
print("-" * 75)

models_list = [
    ("Baseline", baseline, "High Bias"),
    ("Logistic Reg", lr_pipeline, "Good Balance"),
    ("KNN (K=1)", Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(1))]), "High Variance"),
    ("KNN (K=50)", Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(50))]), "High Bias"),
]

for name, model, diag in models_list:
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc
    print(f"{name:<20} {train_acc:<12.3f} {test_acc:<12.3f} {gap:<8.3f} {diag:<20}")

print(f"\n📊 BIAS-VARIANCE DIAGNOSTIC:")
print(f"  Train ≈ Test & both low → High Bias")
print(f"  Train >> Test → High Variance")
print(f"  Train ≈ Test & both high → Good Fit")

# ============================================================================
# ASSIGNMENT 12: DECISION TREES
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 12: DECISION TREE MODEL")
print("="*80)

print(f"\nDecision Trees: Recursive partitioning via binary questions")
print(f"'Is feature X < threshold Y?'\n")

# Test different depths
print(f"--- Decision Tree Performance Across Depths ---\n")
print(f"{'Depth':<6} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<8} {'F1':<8}")
print("-" * 50)

for depth in [2, 4, 6, 8, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    gap = train_acc - test_acc
    f1 = f1_score(y_test, dt.predict(X_test), zero_division=0)
    
    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:<6} {train_acc:<12.3f} {test_acc:<12.3f} {gap:<8.3f} {f1:<8.3f}")

# Final decision tree
dt_final = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=5)
dt_final.fit(X_train, y_train)
dt_pred = dt_final.predict(X_test)

print(f"\nDECISION TREE (depth=4) PERFORMANCE:")
print(f"  Train Accuracy: {dt_final.score(X_train, y_train):.3f}")
print(f"  Test Accuracy:  {dt_final.score(X_test, y_test):.3f}")
print(f"  Test F1:        {f1_score(y_test, dt_pred, zero_division=0):.3f}")

# Feature importance
print(f"\nFEATURE IMPORTANCE (from splits):")
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': dt_final.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    print(f"  {row['Feature']:<20}: {row['Importance']:>7.3f}")

# ============================================================================
# ASSIGNMENT 13: FEATURE IMPORTANCE & PERMUTATION IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 13: FEATURE IMPORTANCE")
print("="*80)

# Train Random Forest for more stable importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"\nMEAN DECREASE IN IMPURITY (MDI) - from Random Forest:")
mdi_df = pd.DataFrame({
    'Feature': feature_columns,
    'MDI': rf.feature_importances_
}).sort_values('MDI', ascending=False)

for _, row in mdi_df.iterrows():
    print(f"  {row['Feature']:<20}: {row['MDI']:>7.3f}")

# Permutation importance
print(f"\nPERMUTATION IMPORTANCE (more reliable):")
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'Feature': feature_columns,
    'Permutation': result.importances_mean,
    'Std': result.importances_std
}).sort_values('Permutation', ascending=False)

for _, row in perm_df.iterrows():
    print(f"  {row['Feature']:<20}: {row['Permutation']:>7.3f} ± {row['Std']:.3f}")

print(f"\nRANDOM FOREST PERFORMANCE:")
print(f"  Test Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(f"  Test Recall:   {recall_score(y_test, rf_pred, zero_division=0):.1%}")
print(f"  Test F1:       {f1_score(y_test, rf_pred, zero_division=0):.3f}")

# ============================================================================
# ASSIGNMENT 14: GRIDSEARCHCV HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 14: GRIDSEARCHCV HYPERPARAMETER TUNING")
print("="*80)

print(f"\nSearching for optimal hyperparameters using cross-validation...\n")

# Grid for Decision Tree
param_grid_dt = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_leaf': [1, 5, 10, 20]
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring='f1',
    return_train_score=True,
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

print(f"DECISION TREE GRID SEARCH RESULTS:\n")
print(f"Best Parameters:  {grid_dt.best_params_}")
print(f"Best CV F1 Score: {grid_dt.best_score_:.3f}")

# Get top configurations
results_df = pd.DataFrame(grid_dt.cv_results_)
top_configs = results_df.nsmallest(5, 'rank_test_score')[[
    'param_max_depth', 'param_min_samples_leaf',
    'mean_train_score', 'mean_test_score', 'std_test_score'
]]

print(f"\nTop 5 Configurations:\n")
print(f"{'Depth':<6} {'Min Leaf':<10} {'Train F1':<10} {'Test F1':<10} {'Std':<8}")
print("-" * 50)
for _, row in top_configs.iterrows():
    print(f"{int(row['param_max_depth']):<6} {int(row['param_min_samples_leaf']):<10} "
          f"{row['mean_train_score']:<10.3f} {row['mean_test_score']:<10.3f} "
          f"{row['std_test_score']:<8.3f}")

# Evaluate best model
best_dt = grid_dt.best_estimator_
best_pred = best_dt.predict(X_test)

print(f"\nTUNED DECISION TREE TEST PERFORMANCE:")
print(f"  Accuracy: {accuracy_score(y_test, best_pred):.3f}")
print(f"  Recall:   {recall_score(y_test, best_pred, zero_division=0):.1%}")
print(f"  F1:       {f1_score(y_test, best_pred, zero_division=0):.3f}")

# Grid for KNN
param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 10, 15],
    'knn__weights': ['uniform', 'distance']
}

grid_knn = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    param_grid_knn,
    cv=5,
    scoring='f1',
    return_train_score=True,
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)

print(f"\nKNN GRID SEARCH RESULTS:\n")
print(f"Best Parameters:  {grid_knn.best_params_}")
print(f"Best CV F1 Score: {grid_knn.best_score_:.3f}")

best_knn = grid_knn.best_estimator_
best_knn_pred = best_knn.predict(X_test)

print(f"\nTUNED KNN TEST PERFORMANCE:")
print(f"  Accuracy: {accuracy_score(y_test, best_knn_pred):.3f}")
print(f"  Recall:   {recall_score(y_test, best_knn_pred, zero_division=0):.1%}")
print(f"  F1:       {f1_score(y_test, best_knn_pred, zero_division=0):.3f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY: ADVANCED ML TOPICS (ASSIGNMENTS 7-14)")
print("="*80)

print(f"""
✅ ASSIGNMENT 7: PRECISION & RECALL
   • Precision: When model flags fraud, how often is it right?
   • Recall: Of all fraud, what % does model catch?
   • Baseline Recall: {baseline_rec:.1%} (catches NO fraud)
   • Model Recall: {lr_rec:.1%} (catches most fraud)

✅ ASSIGNMENT 8: F1-SCORE
   • Harmonic mean of Precision and Recall
   • Baseline F1: {baseline_f1:.3f} (useless)
   • Model F1: {lr_f1:.3f} (actually detects fraud)

✅ ASSIGNMENT 9: CONFUSION MATRIX
   • TP: {tp} (caught fraud)
   • FN: {fn} (missed fraud)
   • FP: {fp} (false alarms)
   • TN: {tn} (correct legitimate)

✅ ASSIGNMENT 10: K-NEAREST NEIGHBORS
   • Non-parametric, instance-based learner
   • Requires feature scaling (done in pipeline)
   • K=5: {accuracy_score(y_test, knn_pred):.3f} accuracy

✅ ASSIGNMENT 11: BIAS-VARIANCE TRADE-OFF
   • High Bias (underfitting): Simple model, both train/test low
   • High Variance (overfitting): Complex model, train >> test
   • Goal: Find balance for good generalization

✅ ASSIGNMENT 12: DECISION TREES
   • Recursive partitioning via binary splits
   • Depth controls complexity (bias-variance)
   • Interpretable: Each prediction is explainable

✅ ASSIGNMENT 13: FEATURE IMPORTANCE
   • MDI: How much did feature reduce impurity?
   • Permutation: How much did shuffling hurt performance?
   • Permutation importance is more reliable

✅ ASSIGNMENT 14: GRIDSEARCHCV
   • Systematically searches hyperparameter combinations
   • Uses cross-validation on training data only
   • Best DT params: {grid_dt.best_params_}
   • Best KNN params: {grid_knn.best_params_}
""")

print("="*80)
print("✓ ALL ADVANCED ASSIGNMENTS (7-14) COMPLETED SUCCESSFULLY!")
print("="*80 + "\n")
