# Telecom Churn Prediction

This repository contains code and models for predicting customer churn in the telecommunications industry.  The project utilizes various machine learning and statistical techniques to identify customers at risk of leaving, enabling proactive retention strategies.

# Summary

The **eda.ipynb** notebook performs an in-depth **Exploratory Data Analysis (EDA)** on a telecom customer churn dataset to identify patterns, anomalies, and key factors influencing customer churn. The findings are:

 * **Service Agreements:** Month‑to‑month customers churn more than those on annual/bi‑annual plans.

 * **Tenure:** Newer customers show higher churn risk.
 
 * **Charges:** Higher MonthlyCharges and abrupt increases signal churn.
 
 * **Services:** Customers without multiple services (e.g., no internet or security add‑ons) are more likely to churn.
 
 * **Payment Methods:** Electronic check users exhibit elevated churn rates.


# Demo

[🔗 View the Streamlit App](https://telecom-churn-prediction-amama.streamlit.app/)


# 📈 Model Performance During Training (AUC-ROC)

| Model                 | CV Mean AUC (± std)        | Test AUC |
| --------------------- | -------------------------- | -------- |
| **CatBoost**          | 0.8511 ± 0.0146            | 0.8472   |
| **CatBoost (tuned)**  | (best trial only) — 0.8385 | —        |
| **XGBoost**           | 0.8503 ± 0.0147            | 0.8481   |
| **XGBoost (tuned)**   | (best trial only) — 0.8407 | —        |
| **LightGBM**          | 0.8482 ± 0.0154            | 0.8506   |
| **LightGBM (tuned)**  | (best trial only) — 0.8383 | —        |
| **Stacking Ensemble** | —                          | 0.8491   |



# 🔎 Streamlit App Evaluation

## 🟦 CatBoost


- **Accuracy:** 81.23%
- **Confusion Matrix:**


|                | **Predicted No** | **Predicted Yes** |
| -------------- | ---------------- | ----------------- |
| **Actual No**  | 4697             | 477               |
| **Actual Yes** | 845              | 1024              |



## 🟨 XGBoost
- **Accuracy:** 81.58%
- **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 4,733        | 441           |
| **Actual Yes** | 856          | 1,013         |

## 🟩 LightGBM
- **Accuracy:** 81.56%
- **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 4,725        | 449           |
| **Actual Yes** | 850          | 1,019         |

## 🟩 Stacking Ensemble
- **Accuracy:** 81.16%
- **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 4,776        | 398           |
| **Actual Yes** | 929          | 940           |


# 📂 Dataset Description
The dataset consists of information about 7,043 telecom customers, each represented by 21 features. It captures demographic information, account details, service usage patterns, and whether or not the customer has churned (i.e., left the company).

## Feature Overview

| Column             | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `customerID`       | Unique customer identifier                                               |
| `gender`           | Customer's gender: Male or Female                                        |
| `SeniorCitizen`    | Indicates if the customer is a senior citizen (1 = Yes, 0 = No)          |
| `Partner`          | Whether the customer has a partner                                       |
| `Dependents`       | Whether the customer has dependents                                      |
| `tenure`           | Number of months the customer has stayed with the company                |
| `PhoneService`     | Indicates if the customer has a phone service                            |
| `MultipleLines`    | Whether the customer has multiple lines                                  |
| `InternetService`  | Type of internet service: DSL, Fiber optic, or None                      |
| `OnlineSecurity`   | Whether the customer has online security add-on                          |
| `OnlineBackup`     | Whether the customer has online backup add-on                            |
| `DeviceProtection` | Whether the customer has device protection add-on                        |
| `TechSupport`      | Whether the customer has tech support add-on                             |
| `StreamingTV`      | Whether the customer has streaming TV                                    |
| `StreamingMovies`  | Whether the customer has streaming movies                                |
| `Contract`         | Type of contract: Month-to-month, One year, or Two year                  |
| `PaperlessBilling` | Whether billing is paperless                                             |
| `PaymentMethod`    | Customer’s payment method (e.g., Electronic check, Bank transfer)        |
| `MonthlyCharges`   | The amount charged to the customer monthly                               |
| `TotalCharges`     | The total amount charged to the customer (as a string; needs conversion) |
| `Churn`            | Whether the customer churned (Yes or No)                                 |





# Exploratory Data Analysis

## 1. Descriptive Analaysis

  * Analysis of categorical features
  * Handling duplicates
  * Analysis of unique values

## 2. Data Wrangling

  * Handling missing values
  * Feature Binning

## 3. Univariate Analysis

  * Statistical Normality Test
      * D'Agostino-Pearson Test
      * Anderson-Darling Test
  * Individual visualization of features


## 4. Bivariate Analysis

  * Numerical & Numerical Correlations
      * Spearman rank-order correlation
  * Numerical & Categorical Correlations
      * Kendall's Tau
      * Mann-Whitney U Test
      * Polytomous with Numeric
  * Dichotomous Correlations
      * Phi's Correlation
      * 
  * Categorical & Categorical Correlations
      * Chi-Sqaure For Independece
      * Cramer's V
      * Uncertainty Coefficient
  * Collinearity
  * Visualization of features in groups of two

## 6. Multivariate Analysis

  * Multicollinearity
  * Frequency Distribution

## 7. Feature Engineering

  * Hot encoding
    
## 8. Data Preparation

  * Train and test data split
  * Encoding and Scaling

## 9. Training of Models

  * Catboost training
  * XGBoost training
  * LGBM training
  * Stack Ensemble training








