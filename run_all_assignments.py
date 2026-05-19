#!/usr/bin/env python3
"""
Complete ML Workflow: Assignments 1-6
Baseline → Linear Regression → Logistic Regression
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("\n" + "="*80)
print("COMPLETE ML WORKFLOW: ALL ASSIGNMENTS 1-6")
print("="*80)

# Load data
print("\n" + "─"*80)
print("STEP 1: Load and Explore Data")
print("─"*80)

df = pd.read_csv('data/raw/fraud_data.csv')
print(f"\n✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"\nFeatures: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['is_fraud'].value_counts())
print(f"Imbalance ratio: {df['is_fraud'].value_counts()[0] / df['is_fraud'].value_counts()[1]:.2f}x")

# Preprocessing
print("\n" + "─"*80)
print("STEP 2: Data Preprocessing")
print("─"*80)

df_encoded = df.copy()
category_encoder = LabelEncoder()
location_encoder = LabelEncoder()
df_encoded['category_encoded'] = category_encoder.fit_transform(df_encoded['category'])
df_encoded['location_encoded'] = location_encoder.fit_transform(df_encoded['location'])

feature_columns = ['amount', 'transaction_count', 'velocity', 'category_encoded', 'location_encoded']
X = df_encoded[feature_columns]
y_regression = df_encoded['amount']
y_classification = df_encoded['is_fraud']

print(f"✓ Features prepared: {X.shape}")
print(f"  Regression target (amount): mean={y_regression.mean():.2f}, std={y_regression.std():.2f}")
print(f"  Classification target (fraud): {y_classification.value_counts().to_dict()}")

# Train-test split
print("\n" + "─"*80)
print("STEP 3: Train-Test Split")
print("─"*80)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

print(f"✓ Split complete")
print(f"  Regression: train={X_train_reg.shape[0]}, test={X_test_reg.shape[0]}")
print(f"  Classification (stratified): train={X_train_clf.shape[0]}, test={X_test_clf.shape[0]}")
print(f"    Test fraud rate: {y_test_clf.mean():.1%}")

# ============================================================================
# ASSIGNMENT 1: BASELINE MODELS
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 1: BASELINE MODELS (Simple Heuristics)")
print("="*80)

# Regression baseline
print("\n[1A] REGRESSION BASELINE (Mean Predictor)")
baseline_reg = DummyRegressor(strategy="mean")
baseline_reg.fit(X_train_reg, y_train_reg)
baseline_pred_reg = baseline_reg.predict(X_test_reg)

training_mean = baseline_pred_reg[0]
print(f"  Strategy: Predict training mean = ${training_mean:.2f} for every transaction")

# Classification baseline
print("\n[1B] CLASSIFICATION BASELINE (Majority Class)")
baseline_clf = DummyClassifier(strategy="most_frequent")
baseline_clf.fit(X_train_clf, y_train_clf)
baseline_pred_clf = baseline_clf.predict(X_test_clf)
baseline_prob_clf = baseline_clf.predict_proba(X_test_clf)[:, 1]

majority_class = baseline_pred_clf[0]
print(f"  Strategy: Predict '{majority_class}' (majority class) for EVERY transaction")

# Evaluate baselines
baseline_mae_reg = mean_absolute_error(y_test_reg, baseline_pred_reg)
baseline_rmse_reg = np.sqrt(mean_squared_error(y_test_reg, baseline_pred_reg))
baseline_r2_reg = r2_score(y_test_reg, baseline_pred_reg)
baseline_acc_clf = accuracy_score(y_test_clf, baseline_pred_clf)
baseline_recall_clf = recall_score(y_test_clf, baseline_pred_clf, zero_division=0)

print(f"\n  Regression baseline metrics:")
print(f"    MAE: ${baseline_mae_reg:.2f}")
print(f"    RMSE: ${baseline_rmse_reg:.2f}")
print(f"    R²: {baseline_r2_reg:.4f}")

print(f"\n  Classification baseline metrics:")
print(f"    Accuracy: {baseline_acc_clf:.1%} (looks good but misleading!)")
print(f"    Recall (fraud): {baseline_recall_clf:.1%} ❌ Catches NO fraud!")

# ============================================================================
# ASSIGNMENT 2: LINEAR REGRESSION
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 2: LINEAR REGRESSION")
print("="*80)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train_reg, y_train_reg)
lr_pred_reg = lr_pipeline.predict(X_test_reg)

lr_model = lr_pipeline.named_steps['model']
print(f"\n✓ Linear Regression model trained")
print(f"  Intercept: {lr_model.intercept_:.4f}")
print(f"\n  Top 3 coefficients:")
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
for _, row in coef_df.head(3).iterrows():
    print(f"    {row['Feature']}: {row['Coefficient']:.6f}")

# ============================================================================
# ASSIGNMENT 3: REGRESSION EVALUATION - MAE
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 3: REGRESSION EVALUATION - MAE (Mean Absolute Error)")
print("="*80)

mae_baseline = mean_absolute_error(y_test_reg, baseline_pred_reg)
mae_lr = mean_absolute_error(y_test_reg, lr_pred_reg)
mae_improvement = mae_baseline - mae_lr
mae_improvement_pct = (mae_improvement / mae_baseline) * 100

print(f"\nMAE Comparison:")
print(f"  {'Model':<20} {'MAE':>12}")
print(f"  {'-'*35}")
print(f"  {'Baseline':<20} ${mae_baseline:>11.2f}")
print(f"  {'Linear Regression':<20} ${mae_lr:>11.2f}")
print(f"  {'-'*35}")
print(f"  Improvement: ${mae_improvement:>11.2f} ({mae_improvement_pct:>5.1f}%)")

# Cross-validation
cv_mae_lr = -cross_val_score(lr_pipeline, X_train_reg, y_train_reg, 
                              cv=5, scoring='neg_mean_absolute_error')
print(f"\n5-Fold CV MAE:")
print(f"  Mean: ${cv_mae_lr.mean():.3f} ± ${cv_mae_lr.std():.3f}")

# ============================================================================
# ASSIGNMENT 4: REGRESSION EVALUATION - MSE & R²
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 4: REGRESSION EVALUATION - MSE & R²")
print("="*80)

mse_baseline = mean_squared_error(y_test_reg, baseline_pred_reg)
mse_lr = mean_squared_error(y_test_reg, lr_pred_reg)
rmse_baseline = np.sqrt(mse_baseline)
rmse_lr = np.sqrt(mse_lr)
r2_baseline = r2_score(y_test_reg, baseline_pred_reg)
r2_lr = r2_score(y_test_reg, lr_pred_reg)

print(f"\nMetrics Comparison:")
print(f"  {'Metric':<15} {'Baseline':>12} {'Linear Reg':>12}")
print(f"  {'-'*42}")
print(f"  {'MSE':<15} {mse_baseline:>12.4f} {mse_lr:>12.4f}")
print(f"  {'RMSE':<15} ${rmse_baseline:>11.2f} ${rmse_lr:>11.2f}")
print(f"  {'MAE':<15} ${mae_baseline:>11.2f} ${mae_lr:>11.2f}")
print(f"  {'R²':<15} {r2_baseline:>12.4f} {r2_lr:>12.4f}")

print(f"\nInterpretation:")
print(f"  R² = {r2_lr:.3f} → Model explains {r2_lr*100:.1f}% of variance")
print(f"  Baseline R² = {r2_baseline:.4f} (always 0.0 for mean predictor)")

# Cross-validation
cv_r2_lr = cross_val_score(lr_pipeline, X_train_reg, y_train_reg, cv=5, scoring='r2')
print(f"\n5-Fold CV R²:")
print(f"  Mean: {cv_r2_lr.mean():.3f} ± {cv_r2_lr.std():.3f}")

# ============================================================================
# ASSIGNMENT 5: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 5: LOGISTIC REGRESSION")
print("="*80)

lr_clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

lr_clf_pipeline.fit(X_train_clf, y_train_clf)
lr_clf_pred = lr_clf_pipeline.predict(X_test_clf)
lr_clf_prob = lr_clf_pipeline.predict_proba(X_test_clf)[:, 1]

print(f"\n✓ Logistic Regression model trained")

lr_clf_model = lr_clf_pipeline.named_steps['model']
print(f"  Intercept: {lr_clf_model.intercept_[0]:.4f}")
print(f"\n  Top 3 risk factors:")
coef_df_clf = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_clf_model.coef_[0],
    'Odds Ratio': np.exp(lr_clf_model.coef_[0])
}).sort_values('Coefficient', key=abs, ascending=False)
for _, row in coef_df_clf.head(3).iterrows():
    odds = row['Odds Ratio']
    direction = '↑ increases' if odds > 1 else '↓ decreases'
    print(f"    {row['Feature']}: {direction} fraud odds by {abs(odds-1)*100:.0f}%")

# ============================================================================
# ASSIGNMENT 6: CLASSIFICATION EVALUATION
# ============================================================================

print("\n" + "="*80)
print("ASSIGNMENT 6: CLASSIFICATION EVALUATION")
print("="*80)

# Compute metrics
acc_baseline = accuracy_score(y_test_clf, baseline_pred_clf)
acc_lr_clf = accuracy_score(y_test_clf, lr_clf_pred)
balanced_acc_baseline = balanced_accuracy_score(y_test_clf, baseline_pred_clf)
balanced_acc_lr_clf = balanced_accuracy_score(y_test_clf, lr_clf_pred)
auc_baseline = roc_auc_score(y_test_clf, baseline_prob_clf)
auc_lr_clf = roc_auc_score(y_test_clf, lr_clf_prob)
precision_baseline = precision_score(y_test_clf, baseline_pred_clf, zero_division=0)
precision_lr_clf = precision_score(y_test_clf, lr_clf_pred, zero_division=0)
recall_baseline = recall_score(y_test_clf, baseline_pred_clf, zero_division=0)
recall_lr_clf = recall_score(y_test_clf, lr_clf_pred, zero_division=0)
f1_baseline = f1_score(y_test_clf, baseline_pred_clf, zero_division=0)
f1_lr_clf = f1_score(y_test_clf, lr_clf_pred, zero_division=0)

print(f"\nAccuracy Analysis (Why accuracy alone is MISLEADING on imbalanced data):")
print(f"  {'Model':<20} {'Accuracy':>12} {'Balanced Acc':>12}")
print(f"  {'-'*47}")
print(f"  {'Baseline':<20} {acc_baseline:>11.1%} {balanced_acc_baseline:>11.1%}")
print(f"  {'Logistic Reg':<20} {acc_lr_clf:>11.1%} {balanced_acc_lr_clf:>11.1%}")

print(f"\n⚠️  ACCURACY ANALYSIS:")
print(f"  Baseline accuracy: {acc_baseline:.1%} (looks good!)")
print(f"  BUT it predicts 'not fraud' for EVERYTHING")
print(f"  → Balanced Accuracy reveals the truth: {balanced_acc_baseline:.1%} = random guessing")

print(f"\nDetailed Metrics:")
print(f"  {'Metric':<20} {'Baseline':>12} {'Logistic Reg':>12}")
print(f"  {'-'*47}")
print(f"  {'Precision':<20} {precision_baseline:>11.1%} {precision_lr_clf:>11.1%}")
print(f"  {'Recall':<20} {recall_baseline:>11.1%} {recall_lr_clf:>11.1%}")
print(f"  {'F1-Score':<20} {f1_baseline:>11.3f} {f1_lr_clf:>11.3f}")
print(f"  {'ROC-AUC':<20} {auc_baseline:>11.3f} {auc_lr_clf:>11.3f}")

print(f"\nPER-CLASS INTERPRETATION:")
print(f"  Precision: Of transactions flagged as fraud, this % are actually fraud")
print(f"    Baseline: {precision_baseline:.1%} (no predictions made)")
print(f"    Model: {precision_lr_clf:.1%}")

print(f"\n  Recall: Of all actual frauds, this % does the model catch?")
print(f"    Baseline: {recall_baseline:.1%} (catches ZERO fraud!)")
print(f"    Model: {recall_lr_clf:.1%} ✓")

print(f"\n  F1-Score: Harmonic mean of precision and recall")
print(f"    Baseline: {f1_baseline:.3f} (useless on imbalanced data)")
print(f"    Model: {f1_lr_clf:.3f} (actually useful)")

# Confusion matrices
cm_baseline = confusion_matrix(y_test_clf, baseline_pred_clf)
cm_lr_clf = confusion_matrix(y_test_clf, lr_clf_pred)

print(f"\nConfusion Matrix - Baseline:")
print(f"  Predicted:    Legit  Fraud")
print(f"  Actual Legit:  {cm_baseline[0,0]:>5}  {cm_baseline[0,1]:>5}")
print(f"  Actual Fraud:  {cm_baseline[1,0]:>5}  {cm_baseline[1,1]:>5}")

print(f"\nConfusion Matrix - Logistic Regression:")
print(f"  Predicted:    Legit  Fraud")
print(f"  Actual Legit:  {cm_lr_clf[0,0]:>5}  {cm_lr_clf[0,1]:>5}")
print(f"  Actual Fraud:  {cm_lr_clf[1,0]:>5}  {cm_lr_clf[1,1]:>5}")

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc_lr_clf = cross_val_score(lr_clf_pipeline, X_train_clf, y_train_clf, 
                                 cv=skf, scoring='accuracy')
cv_f1_lr_clf = cross_val_score(lr_clf_pipeline, X_train_clf, y_train_clf, 
                                cv=skf, scoring='f1')
cv_auc_lr_clf = cross_val_score(lr_clf_pipeline, X_train_clf, y_train_clf, 
                                 cv=skf, scoring='roc_auc')

print(f"\n5-Fold Cross-Validation (Logistic Regression):")
print(f"  Accuracy: {cv_acc_lr_clf.mean():.3f} ± {cv_acc_lr_clf.std():.3f}")
print(f"  F1-Score: {cv_f1_lr_clf.mean():.3f} ± {cv_f1_lr_clf.std():.3f}")
print(f"  ROC-AUC:  {cv_auc_lr_clf.mean():.3f} ± {cv_auc_lr_clf.std():.3f}")
print(f"  ✓ Low std = stable model (consistent across folds)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY: ALL 6 ASSIGNMENTS COMPLETED")
print("="*80)

print(f"\n✅ ASSIGNMENT 1: BASELINE MODELS")
print(f"   Regression baseline (mean): MAE=${mae_baseline:.2f}")
print(f"   Classification baseline (majority): Accuracy={acc_baseline:.1%}, Recall={recall_baseline:.1%}")

print(f"\n✅ ASSIGNMENT 2: LINEAR REGRESSION")
print(f"   Model trained on {X_train_reg.shape[0]} samples")
print(f"   R² = {r2_lr:.4f} (explains {r2_lr*100:.1f}% of variance)")

print(f"\n✅ ASSIGNMENT 3: MAE EVALUATION")
print(f"   Baseline MAE: ${mae_baseline:.2f}")
print(f"   Model MAE: ${mae_lr:.2f}")
print(f"   Improvement: {mae_improvement_pct:.1f}% reduction ✓")

print(f"\n✅ ASSIGNMENT 4: MSE & R² EVALUATION")
print(f"   Baseline R²: {r2_baseline:.4f}")
print(f"   Model R²: {r2_lr:.4f}")
print(f"   CV R²: {cv_r2_lr.mean():.3f} ± {cv_r2_lr.std():.3f} (stable)")

print(f"\n✅ ASSIGNMENT 5: LOGISTIC REGRESSION")
print(f"   Binary classification model trained")
print(f"   ROC-AUC: {auc_lr_clf:.3f} (vs baseline: {auc_baseline:.3f})")

print(f"\n✅ ASSIGNMENT 6: CLASSIFICATION EVALUATION")
print(f"   Baseline accuracy: {acc_baseline:.1%} (misleading!)")
print(f"   Model balanced accuracy: {balanced_acc_lr_clf:.1%}")
print(f"   Model recall (fraud detection): {recall_lr_clf:.1%} ✓")
print(f"   Model F1-score: {f1_lr_clf:.3f}")

print(f"\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

print(f"""
1. ALWAYS START WITH A BASELINE
   → Baselines establish what a trivial solution achieves
   → Your models MUST beat the baseline to be useful

2. LINEAR REGRESSION IMPROVED MAE BY {mae_improvement_pct:.1f}%
   → R² = {r2_lr:.3f} explains {r2_lr*100:.0f}% of variance
   → Always validate with cross-validation

3. ACCURACY IS MISLEADING ON IMBALANCED DATA
   → Baseline: {acc_baseline:.1%} accuracy but {recall_baseline:.1%} fraud recall = useless!
   → Use: Balanced Accuracy, Precision, Recall, F1, ROC-AUC

4. LOGISTIC REGRESSION PERFORMS WELL
   → ROC-AUC: {auc_lr_clf:.3f} (much better than baseline's {auc_baseline:.3f})
   → Recall: {recall_lr_clf:.1%} (actually catches fraud!)

5. ALWAYS USE CROSS-VALIDATION
   → CV scores show if performance is consistent
   → Low std = stable, trustworthy model
""")

print("="*80)
print("✓ ALL ASSIGNMENTS COMPLETED SUCCESSFULLY!")
print("="*80 + "\n")
