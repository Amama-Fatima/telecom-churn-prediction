---

# 📞 Telecom Churn Prediction

This repository contains code and models for predicting **customer churn** in the telecommunications industry.
📊 The project uses machine learning and statistical techniques to identify customers at risk of leaving.

---

# 🧠 Summary

The **`eda.ipynb`** notebook performs an in-depth **Exploratory Data Analysis (EDA)** on a telecom churn dataset to uncover:

* 📆 **Service Agreements:** Month-to-month customers churn more than those on annual/bi-annual plans
* ⏳ **Tenure:** Newer customers show higher churn risk
* 💸 **Charges:** High monthly charges and sudden increases are churn signals
* 📡 **Services:** Fewer service add-ons (e.g., no internet/security) = higher churn
* 💳 **Payment Methods:** Customers using *electronic checks* churn more frequently

---

# 🚀 Demo

[▶️ **View the Streamlit App**](https://telecom-churn-prediction-amama.streamlit.app/)

### ⚡ Quick Start:

1. 🔹 Go to the **Demo Data** tab
2. 🔹 Click **Run Prediction**
3. 🔹 (Optional) Click **Evaluate Predictions**

### 🧍 Manual Prediction:

1. 🟢 Go to the **Manual Input** tab
2. 🟢 Fill in customer details
3. 🟢 Click **Predict Churn**

### 📁 Upload Your Data:

1. 📤 Go to the **Upload Data** tab
2. 📤 Upload your CSV file
3. 📤 (Optional) Upload true labels for evaluation

---

# 📈 Model Performance During Training (AUC-ROC)

| ⚙️ Model                 | 🎯 CV Mean AUC (± std) | 🧪 Test AUC |
| ------------------------ | ---------------------- | ----------- |
| 🐱 **CatBoost**          | 0.8511 ± 0.0146        | 0.8472      |
| 🐱 **CatBoost (tuned)**  | (best trial) — 0.8385  | —           |
| ⚔️ **XGBoost**           | 0.8503 ± 0.0147        | 0.8481      |
| ⚔️ **XGBoost (tuned)**   | (best trial) — 0.8407  | —           |
| 💡 **LightGBM**          | 0.8482 ± 0.0154        | 0.8506      |
| 💡 **LightGBM (tuned)**  | (best trial) — 0.8383  | —           |
| 🧠 **Stacking Ensemble** | —                      | 0.8491      |

---

# 📊 Streamlit App Evaluation

## 🟦 CatBoost

* ✅ **Accuracy:** 81.23%
* 🔢 **Confusion Matrix:**

|                | **Predicted No** | **Predicted Yes** |
| -------------- | ---------------- | ----------------- |
| **Actual No**  | 4,697            | 477               |
| **Actual Yes** | 845              | 1,024             |

---

## 🟨 XGBoost

* ✅ **Accuracy:** 81.58%
* 🔢 **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | 4,733        | 441           |
| **Actual Yes** | 856          | 1,013         |

---

## 🟩 LightGBM

* ✅ **Accuracy:** 81.56%
* 🔢 **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | 4,725        | 449           |
| **Actual Yes** | 850          | 1,019         |

---

## 🟩 Stacking Ensemble

* ✅ **Accuracy:** 81.16%
* 🔢 **Confusion Matrix:**

|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | 4,776        | 398           |
| **Actual Yes** | 929          | 940           |

---

# 📂 Dataset Description

This dataset includes **7,043 telecom customers** with 21 features: demographics, service usage, account info, and churn status.

## 🧾 Feature Overview

| 🏷️ Column         | 📝 Description                                     |
| ------------------ | -------------------------------------------------- |
| `customerID`       | Unique customer ID                                 |
| `gender`           | Gender: Male or Female                             |
| `SeniorCitizen`    | 1 = Senior citizen, 0 = Not                        |
| `Partner`          | Whether the customer has a partner                 |
| `Dependents`       | Whether the customer has dependents                |
| `tenure`           | Months with the company                            |
| `PhoneService`     | Phone service subscription                         |
| `MultipleLines`    | Has multiple phone lines                           |
| `InternetService`  | DSL, Fiber optic, or None                          |
| `OnlineSecurity`   | Has online security add-on                         |
| `OnlineBackup`     | Has online backup add-on                           |
| `DeviceProtection` | Has device protection add-on                       |
| `TechSupport`      | Has technical support add-on                       |
| `StreamingTV`      | Has streaming TV                                   |
| `StreamingMovies`  | Has streaming movies                               |
| `Contract`         | Contract type: Month-to-month, One year, Two year  |
| `PaperlessBilling` | Uses paperless billing                             |
| `PaymentMethod`    | Payment type (e.g., Electronic check)              |
| `MonthlyCharges`   | Monthly bill amount                                |
| `TotalCharges`     | Total charged amount (as string; needs conversion) |
| `Churn`            | Whether customer churned: Yes or No                |

---

# 🔍 Exploratory Data Analysis

## 1️⃣ Descriptive Analysis

* 📦 Categorical feature review
* ❌ Duplicate detection
* 🔍 Unique value profiling

## 2️⃣ Data Wrangling

* 🧼 Handle missing values
* 🧱 Feature binning

## 3️⃣ Univariate Analysis

* 🧪 Normality tests

  * D’Agostino-Pearson
  * Anderson-Darling
* 📊 Individual visualizations

## 4️⃣ Bivariate Analysis

* 🔗 Correlations:

  * 📉 Numerical vs Numerical (Spearman)
  * 📋 Categorical vs Numerical (Kendall’s Tau, Mann-Whitney U)
  * 🔘 Dichotomous (Phi Coefficient)
  * 🔀 Categorical vs Categorical (Chi-Square, Cramér’s V, Uncertainty Coefficient)
* 🔄 Collinearity Checks
* 🖼️ Feature Pair Visualizations

## 6️⃣ Multivariate Analysis

* 🧠 Multicollinearity
* 📊 Frequency Distribution

## 7️⃣ Feature Engineering

* 🔥 One-Hot Encoding

## 8️⃣ Data Preparation

* ✂️ Train-Test Split
* 🧮 Encoding + Scaling

## 9️⃣ Model Training

* 🐱 CatBoost
* ⚔️ XGBoost
* 💡 LightGBM
* 🧠 Stacking Ensemble

---
