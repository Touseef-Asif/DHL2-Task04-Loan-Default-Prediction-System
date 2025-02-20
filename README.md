# DHL2-Task04-Loan-Default-Prediction-System

## Project Description
This project focuses on building a predictive model for identifying high-risk loan applicants using the **Lending Club Loan Dataset**. The goal is to help financial institutions reduce defaults by accurately predicting loan defaults based on applicant financial data.

**Key Objectives**:
- Perform exploratory data analysis (EDA) to identify critical features.
- Train and compare multiple ML models (LightGBM, SVM).
- Evaluate models using Precision, Recall, F1 Score, and AUC-ROC curves.
- Generate actionable insights for lenders using SHAP values.

---

## Dataset Overview
**Dataset**: [Lending Club Loan Dataset](https://www.kaggle.com/wordsforthewise/lending-club)  
**Features**:  
- `loan_amnt`: Loan amount
- `term`: Loan term (36 or 60 months)
- `int_rate`: Interest rate
- `grade`: Loan grade (A-G)
- `annual_inc`: Annual income
- `dti`: Debt-to-income ratio
- `revol_util`: Revolving line utilization rate
- `pub_rec`: Number of derogatory public records
- `open_acc`: Number of open credit lines
- `total_acc`: Total number of credit lines
- `mort_acc`: Number of mortgage accounts
- `installment`: Monthly payment
- `income_to_loan_ratio`: Ratio of annual income to loan amount
- `credit_length`: Length of credit history (in years)

**Target Variable**:  
- `default`: Binary indicator (1 = Default, 0 = No Default)

---

## Preprocessing Steps
1. **Handling Missing Values**:
   - Numerical columns: Filled with median values.
   - Categorical columns: Filled with 'Unknown'.

2. **Feature Engineering**:
   - Created `income_to_loan_ratio` and `credit_length` features.
   - Log-transformed skewed features (`annual_inc`, `loan_amnt`, `revol_bal`, `installment`).

3. **Encoding**:
   - Ordinal encoding for `grade` and `sub_grade`.
   - One-hot encoding for categorical variables (`home_ownership`, `verification_status`, `purpose`, `addr_state`).

4. **Feature Scaling**:
   - Standardized numerical features using `StandardScaler`.

5. **Class Imbalance Handling**:
   - Applied SMOTETomek to balance the dataset.

---

## Model Training
### Algorithms Used:
1. **LightGBM**:
   - Optimized hyperparameters using Optuna.
   - Key hyperparameters: `num_leaves`, `learning_rate`, `feature_fraction`, `bagging_fraction`.

2. **SVM**:
   - Used RBF kernel with hyperparameter tuning.

### Evaluation Metrics:
- **Classification Report**: Precision, Recall, F1 Score.
- **ROC AUC Score**: Area under the ROC curve.
- **Matthews Correlation Coefficient (MCC)**: Balanced measure for binary classification.

---

## Results
### Model Performance:
- **LightGBM**:
  - ROC AUC Score: 0.4664
  - F1 Score: 0.82 (weighted)
  - MCC: -0.0015

- **SVM**:
  - ROC AUC Score: 0.45
  - F1 Score: 0.80 (weighted)
  - MCC: -0.0020

### Key Insights:
1. **Top Features Contributing to Default**:
   - `int_rate`: Higher interest rates increase default risk.
   - `dti`: High debt-to-income ratio is a strong predictor of default.
   - `revol_util`: High revolving utilization indicates financial stress.
   - `income_to_loan_ratio`: Low income-to-loan ratio increases default likelihood.
   - `credit_length`: Shorter credit history correlates with higher risk.

2. **General Risk Indicators**:
   - High Debt-to-Income Ratio (DTI > 20).
   - Low FICO Score (FICO < 650).
   - High Revolving Utilization (Utilization > 60%).
   - Large Loan Amounts (Loan > $25k).

---

## Recommendations for Lenders
1. **Stricter DTI Thresholds**:
   - Implement stricter DTI thresholds for high-risk categories (DTI > 20).

2. **Lower Interest Rates**:
   - Offer lower interest rates for borrowers with FICO > 700 to reduce default risk.

3. **Credit Counseling**:
   - Add credit counseling requirements for borrowers with utilization > 75%.

4. **Income-to-Loan Ratio**:
   - Consider income-to-loan ratio as an additional qualification metric.

5. **Limit Large Loans**:
   - Limit large loans for borrowers with low credit scores and high DTI.

---

## Challenges & Solutions
1. **Class Imbalance**:
   - Solved using SMOTETomek to balance the dataset.

2. **High Dimensionality**:
   - Reduced dimensions using PCA and feature selection.

3. **Missing Values**:
   - Aggressively removed high-missing columns and imputed remaining values.

4. **Hyperparameter Tuning**:
   - Used Optuna for efficient hyperparameter optimization.

---

## How to Run the Code
1. **Install Dependencies**:
   ```bash
   pip install lightgbm imbalanced-learn shap optuna
