# SyriaTel Customer Churn Analysis

**Author:** Joel Muoki Andrew 

**Date:** April 2026  

## Overview

This project documents a full end-to-end data science workflow for predicting customer churn at SyriaTel, a telecommunications company. Customer churn refers to the phenomenon where customers stop using a company's services. Identifying customers likely to churn before they do allow businesses to take proactive retention action, saving significant revenue.

---

## Business & Data Understanding

**Stakeholder:** SyriaTel Customer Retention Team

**Business Question:** Are there predictable patterns in customer behavior that indicate someone is about to churn?

The dataset contains 3,333 customer records and 20 features covering account information, usage patterns (day, evening, night, international), plan subscriptions, and customer service call history. The target variable is `churn` (True/False). Only 14.5% of customers churned, making this an imbalanced classification problem — addressed using `class_weight='balanced'` across all models.

---

## Modeling

Four models were built iteratively, starting simple and increasing in complexity:

1. **Logistic Regression** — interpretable baseline; good recall but limited by linear decision boundary
2. **Decision Tree (Default)** — captures non-linear patterns but overfits significantly (depth 18)
3. **Decision Tree (Tuned)** — constrained via GridSearchCV; better generalization
4. **Random Forest (Tuned)** — final model; ensemble of 200 trees, highest ROC-AUC and F1-Score

**Primary metric: Recall** — missing a churner (False Negative) costs far more than a false alarm.

---

## Evaluation

| Model | Recall | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7320 | 0.4718 | 0.8153 |
| Decision Tree (Default) | 0.6289 | 0.6256 | 0.7820 |
| Decision Tree (Tuned) | 0.7010 | 0.5714 | 0.8549 |
| **Random Forest (Final)** | **0.7423** | **0.6890** | **0.8701** |

The Random Forest was selected as the final model for its best overall balance of F1-Score and ROC-AUC, with significantly reduced overfitting compared to the single Decision Tree.

Top churn drivers identified: **customer service calls**, **total day charge**, and **international plan** subscription.

---

## Conclusion

Customers who make 3+ service calls, subscribe to the international plan, or have high day usage charges are at the highest risk of churning. The model can be deployed monthly to score all active customers and flag those above a 60% predicted churn probability for proactive retention outreach.

---

## Repository Structure

```
├── SyriaTel_Churn_Analysis.ipynb       # Main notebook — full analysis
├── Presentation.pdf                    # Non-technical slide deck
├── bigml_59c28831336c6604c800002a.csv  # Dataset
└── README.md                           # This file
```

---

## Requirements

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```
