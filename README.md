# (IN PROGRESS)💳 Credit Card Fraud Detection | Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)
![Libraries](https://img.shields.io/badge/Libraries-pandas%2C%20scikit--learn%2C%20imbalanced--learn%2C%20matplotlib%2C%20seaborn-green)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 🧠 Project Overview

This project applies **machine learning** techniques to detect fraudulent credit card transactions within a highly imbalanced dataset.  
By leveraging **sampling strategies** (undersampling & oversampling), **scaling optimization**, and **model comparison**, the analysis identifies the most effective fraud detection model.

📌 **Goal:** Maximize **recall** and **ROC-AUC** while maintaining strong precision to minimize false positives.  

---

## 📂 Dataset Description

- **Source:** Publicly available, anonymized credit card transaction dataset.  
- **Shape:** 284,807 rows × 31 columns  
- **Target Variable:** Class 
  - `0` → Legitimate transactions  
  - `1` → Fraudulent transactions  
- **Features:**
  - `V1`–`V28`: PCA-transformed numerical features
  - `Time`, `Amount`: Non-PCA features requiring scaling  

---

## ⚙️ Data Preprocessing

### 1. Exploratory Data Analysis (EDA)
- Confirmed **no missing values**.  
- Fraudulent cases represent **~0.172%** of total records.  
- Correlation heatmaps verified orthogonality of PCA features.  
- Visualized the distribution of all variables and performed frequency analysis of fraud vs non-fraud classes

### 2. Feature Scaling
| Feature | Scaler Used | Justification |
|----------|--------------|----------------|
| `Amount` | `RobustScaler` | Handles outliers and skew |
| `Time` | `MinMax` | Scales to a fixed range [0, 1] |
| `V1–V28` | None | Already scaled via PCA |

> PCA inherently standardizes data before component extraction, so additional scaling is unnecessary.

### 3. Train–Test Split
- Split: **80% training / 20% testing**
- Stratified to preserve class proportions (for base model evaluation)
- Fixed `random_state` for reproducibility  

### 4. Handling Class Imbalance
Two rebalancing strategies were used:
- **Random Undersampling** → Downsampled the majority (non-fraud) class.  
- **SMOTE Oversampling** → Generated synthetic fraud samples to achieve a balanced dataset.  

---

## 🤖 Machine Learning Models

| Model | Sampling Strategy | Purpose |
|--------|------------------|----------|
| Logistic Regression | Base / Undersampled / Oversampled | Linear baseline |
| Decision Tree Classifier | Base / Undersampled / Oversampled | Captures nonlinear relationships |
| Random Forest Classifier | Base / Undersampled / Oversampled | Ensemble learner to reduce variance |
| Support Vector Machine (LinearSVC) | Base / Undersampled / Oversampled | Maximizes class separation margin |

---

## 📈 Evaluation Metrics
Each model was evaluated using:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

> 🎯 *High recall and ROC-AUC are prioritized since undetected fraud (false negatives) is more costly than false positives.*

---

## 🧩 Model Performance Summary

### ⚖️ Base (Imbalanced Data)
| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------------|----------------|------------|----------|----------|-----------|
| Logistic Regression | 99.91 | 99.91 | 84.62 | 57.89 | 68.75 | **0.956** |
| Decision Tree | 99.98 | 99.92 | 80.72 | 70.53 | 75.28 | 0.868 |
| Random Forest | 100.00 | 99.95 | 97.10 | 70.53 | 81.71 | 0.924 |
| Linear SVC | 99.92 | 99.91 | 79.01 | 67.37 | 72.73 | 0.952 |

---

### 🔽 Undersampling (Balanced Data)
| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------------|----------------|------------|----------|----------|-----------|
| Logistic Regression | 95.42 | 95.79 | 98.02 | 94.29 | 96.12 | **0.9835** |
| Decision Tree | 96.01 | 95.26 | 98.98 | 92.38 | 95.57 | 0.966 |
| Random Forest | 99.89 | 94.74 | 98.97 | 91.43 | 95.05 | 0.984 |
| Linear SVC | 96.02 | 95.79 | 98.02 | 94.29 | 96.12 | 0.974 |

---

### 🔼 Oversampling (SMOTE – Balanced)
| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------------|----------------|------------|----------|----------|-----------|
| Logistic Regression | 94.69 | 94.73 | 97.52 | 91.83 | 94.59 | 98.88 |
| Decision Tree | 93.96 | 93.96 | 96.61 | 91.16 | 93.81 | 98.08 |
| SVM (Linear) | 94.29 | 94.39 | 97.69 | 90.97 | 94.21 | 98.88 |
| Random Forest | **100.00** | **99.99** | **99.99** | **100.00** | **99.99** | **100.00** |

---

## 🏆 Key Insights

- **Balanced Oversampled Random Forest** achieved **ROC-AUC = 1.00** and perfect recall, meaning it identified every fraudulent transaction in the test set.  
- **Balanced Oversampled Logistic Regression** followed closely (ROC-AUC ≈ 0.9888), maintaining strong interpretability.  
- **Undersampling** provided competitive but slightly lower recall.  
- Models trained on the imbalanced dataset had deceptively high accuracy but low fraud detection sensitivity.  
- Class balancing significantly improved **recall** and **AUC**, the two most critical metrics in fraud detection.

> 🧩 **Conclusion:** The best overall model was the **Random Forest (SMOTE-balanced)**, achieving near-perfect classification while avoiding false negatives.

---

## 🧪 Reproducibility

### 🔧 Requirements
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

▶️ Run the Project

# Clone the repository
git clone https://github.com/kodexl/CreditCardFraudDetection.git

# Launch the notebook
jupyter notebook CreditCardFraudDetection.ipynb
Run cells sequentially to reproduce preprocessing, sampling, training, and evaluation results.

📊 Visualizations (Recommended)
Visualization	Description
ROC Curves	Comparison of model discrimination across resampling methods
Confusion Matrices	Fraud detection vs false positives
Feature Distributions	Scaling effect on Amount and Time
Class Distribution	Before vs. after SMOTE

(Add generated plots to /images and link here for GitHub rendering.)
