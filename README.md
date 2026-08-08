# Disaster Risk Classification Using Apache Spark ML

## Machine Learning-Based Classification of Disaster Risk

This project develops an end-to-end machine learning workflow using **Apache Spark and PySpark ML** to classify disaster risk categories from key disaster risk indicators.

The analysis uses four predictor variables:

- Exposure
- Susceptibility
- Coping Capacity
- Adaptive Capacity

The target variable is the **disaster risk category**, classified into:

- Low
- Medium
- High
- Very High

The project covers data preparation, exploratory data analysis, feature engineering, model development, model evaluation, and interpretation of machine learning results.

---

## Project Overview

Disaster risk is influenced by multiple interacting dimensions, including exposure to hazards, susceptibility of populations and systems, coping capacity, and adaptive capacity.

This project applies machine learning to these indicators to develop a multiclass classification framework for disaster risk assessment.

The primary objective is to:

1. Explore the distribution and relationships among disaster risk indicators.
2. Prepare the data for machine learning using PySpark.
3. Develop multiple classification models.
4. Compare model performance using standard evaluation metrics.
5. Identify the most influential predictors using Random Forest feature importance.
6. Evaluate classification performance using a confusion matrix.

---

## Machine Learning Workflow

```text
Disaster Risk Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Preparation & Validation
        │
        ▼
Exploratory Data Analysis
        │
        ├── Risk Category Distribution
        ├── Feature Distributions
        ├── Correlation Analysis
        ├── Boxplots
        └── Exposure vs WRI Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Dataset
        │
        ▼
Train-Test Split
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
Logistic Regression  Decision Tree   Random Forest
        │               │                │
        └───────────────┴────────────────┘
                        │
                        ▼
                Model Evaluation
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Model Comparison      Confusion Matrix
                                   │
                                   ▼
                           Feature Importance
