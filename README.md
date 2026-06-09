# Customer Churn Predictor

## Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Losing customers directly impacts revenue and growth.

This project uses Machine Learning to predict whether a customer is likely to churn based on service usage, billing patterns, contract type, and customer interaction history.

The project includes:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Cross Validation
* ROC-AUC Evaluation
* Feature Importance Analysis
* Confusion Matrix Visualization
* Customer Churn Prediction

---

## Project Structure

```text
project1_churn_predictor/
│
├── data/
│   └── churn_data.csv
│
├── models/
│   └── best_churn_model.pkl
│
├── plots/
│   ├── eda_overview.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── src/
│   └── churn_predictor.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

### Data Generation

Synthetic telecom customer data is generated with realistic customer behavior patterns including:

* Contract Type
* Monthly Charges
* Total Charges
* Support Calls
* Late Payments
* Internet Services
* Number of Products

---

### Feature Engineering

Additional features created:

* Average Monthly Spend
* Charges Per Product
* Customer Value
* Service Usage Score

---

### Models Used

| Model               | Purpose                      |
| ------------------- | ---------------------------- |
| Logistic Regression | Baseline Linear Classifier   |
| Random Forest       | Ensemble Learning            |
| Gradient Boosting   | Sequential Boosting Ensemble |

---

## Model Performance

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.735   |
| Random Forest       | 0.730   |
| Gradient Boosting   | 0.710   |

Best Model: **Logistic Regression**

---

## Exploratory Data Analysis

### EDA Dashboard

![EDA Overview](plots/eda_overview.png)

---

## ROC Curve Comparison

![ROC Curves](plots/roc_curves.png)

---

## Feature Importance

![Feature Importance](plots/feature_importance.png)

---

## Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

---

## Example Prediction

```python
sample_customer = {
    "tenure_months": 5,
    "monthly_charges": 95.0,
    "num_products": 1,
    "support_calls": 5,
    "late_payments": 2
}
```

Output:

```text
Churn Probability: 61.87%
Decision: CHURN
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

## Future Improvements

* Streamlit Dashboard
* FastAPI Deployment
* SHAP Explainability
* Real Telecom Dataset Integration
* Hyperparameter Optimization

---

## Skills Demonstrated

Machine Learning • Data Analysis • Feature Engineering • Model Evaluation • Classification • Cross Validation • Data Visualization • Python Development
