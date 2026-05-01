# Understanding Supervised Learning Problem Types

This directory contains the submission for the "Understanding Supervised Learning Problem Types" assignment.

## Scenario Analysis

### Scenario 1
**Description:** A fintech company wants to predict whether a credit card transaction is fraudulent based on transaction amount, time, merchant category, and customer history.
- **Problem Type:** Binary Classification
- **Target Variable:** Fraudulent (Yes = 1, No = 0)
- **Metric Justification:** Precision, Recall, F1-Score, and ROC-AUC. Since fraud detection is typically highly imbalanced (legitimate transactions vastly outnumber fraudulent ones), accuracy is misleading. Recall is critical to catch as many frauds as possible, while precision ensures we don't block too many legitimate transactions. F1-score balances both.
- **Algorithm Families:** Logistic Regression, Random Forest, Gradient Boosting (XGBoost/LightGBM).
- **Why alternative problem types would be incorrect:** Treating this as regression makes no sense since we are not predicting a continuous quantity (like "how fraudulent" it is on an unbounded scale). Using multi-class would be incorrect as there are only two mutually exclusive outcomes.

### Scenario 2
**Description:** A real estate platform wants to estimate the sale price of houses using size, number of bedrooms, location score, and age of the property.
- **Problem Type:** Regression (Multiple Linear / Non-Linear Regression)
- **Target Variable:** Sale price (Continuous numerical value in dollars)
- **Metric Justification:** MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R². MAE provides the average error in dollars, making it highly interpretable for business. RMSE penalizes larger errors more heavily, which is useful if big mispricings are particularly bad.
- **Algorithm Families:** Linear Regression, Ridge/Lasso, Random Forest Regressor, Gradient Boosting Regressor.
- **Why alternative problem types would be incorrect:** Treating this as classification by binning into "Low/Medium/High" price brackets discards valuable magnitude information (e.g., $300k vs $310k are treated identically or separated completely if they fall on a boundary).

### Scenario 3
**Description:** A streaming platform wants to tag each movie with all relevant genres (Action, Comedy, Drama, Thriller, etc.) based on its script and metadata. A movie can belong to multiple genres simultaneously.
- **Problem Type:** Multi-Label Classification
- **Target Variable:** Set of applicable genres (e.g., [Action=1, Comedy=1, Drama=0...])
- **Metric Justification:** Hamming Loss (fraction of incorrectly predicted labels) and Subset Accuracy (fraction of instances where all labels are predicted perfectly), plus Micro/Macro F1 across labels.
- **Algorithm Families:** One-Vs-Rest strategies (one binary classifier per genre), Neural Networks with sigmoid output for each class.
- **Why alternative problem types would be incorrect:** Multi-class classification would be incorrect because multi-class implies mutually exclusive categories (forcing the model to pick only one genre). Regression is incorrect as genres are discrete categories.

### Scenario 4
**Description:** A retail chain wants to predict how many units of a product will be sold next week in each store.
- **Problem Type:** Count Regression
- **Target Variable:** Number of units sold (Non-negative integer)
- **Metric Justification:** MAE, RMSE, or Poisson Deviance. MAE tells us how many units off we are on average.
- **Algorithm Families:** Poisson Regression, Random Forest Regressor.
- **Why alternative problem types would be incorrect:** Classification is incorrect because sales can take any non-negative integer value, leading to potentially thousands of "classes" and ignoring the ordering/magnitude (selling 10 is closer to selling 11 than 100). Standard continuous regression can sometimes predict negative values, so specialized count regression (Poisson) or tree-based regressors are often best.

### Scenario 5
**Description:** A hospital wants to predict which of three disease categories (Viral, Bacterial, Autoimmune) a patient’s symptoms correspond to. Each patient can belong to only one category.
- **Problem Type:** Multi-Class Classification
- **Target Variable:** Disease category (1 out of 3 possible mutually exclusive states)
- **Metric Justification:** Macro F1 (if we care equally about all diseases regardless of frequency) and Confusion Matrix (to see misclassifications between specific pairs of diseases). Accuracy is okay only if the classes are perfectly balanced.
- **Algorithm Families:** Multinomial Logistic Regression, Random Forest, SVM.
- **Why alternative problem types would be incorrect:** Multi-label would be incorrect because a patient can only belong to one category in this scenario. Regression would be incorrect because encoding the diseases as 1, 2, and 3 would erroneously imply an ordinal relationship (e.g., Bacterial is "between" Viral and Autoimmune).

## Evidence of Correct Metric Usage
Please see `problem_type_analysis.py` for synthetic code examples demonstrating both classification and regression, correctly computing their respective metrics.
