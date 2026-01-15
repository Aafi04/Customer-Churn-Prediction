# 📊 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Keras-orange)

A comprehensive machine learning project for predicting customer churn using multiple algorithms including XGBoost and Neural Networks, with an interactive Streamlit dashboard for visualization and analysis.

---

## 🎯 Problem Statement

In today's competitive business landscape, customer retention is paramount for sustainable growth and success. Our challenge is to develop a predictive model that can identify customers who are at risk of churning – discontinuing their use of our service.

**Objective:** Build a machine learning model that can accurately predict whether a customer is likely to churn based on their historical usage behavior, demographic information, and subscription details.

---

## 📁 Project Structure

```
ML Project/
├── 📓 Customer Churn Prediction.ipynb  # Main analysis notebook
├── 🎛️ app.py                           # Streamlit dashboard
├── 🐍 clean_code.py                    # Python script version
├── 📤 export_metrics.py                # Metrics export utility
├── 🤖 ChurnClassifier.h5               # Saved Keras model
├── 📊 model_metrics.json               # Pre-computed metrics
├── 📈 feature_importances_*.csv        # Feature importance data
├── 📋 requirements.txt                 # Python dependencies
├── 📄 Customer_Churn_Prediction_Report.md  # Detailed project report
└── 📖 README.md                        # This file
```

---

## 📊 Dataset Overview

| Attribute         | Value          |
| ----------------- | -------------- |
| **Total Records** | 10,000         |
| **Features**      | 9              |
| **Target**        | Churn (Binary) |
| **Churn Rate**    | 49.2%          |

### Features

| Feature                    | Type        | Description                      |
| -------------------------- | ----------- | -------------------------------- |
| CustomerID                 | ID          | Unique identifier                |
| Name                       | Text        | Customer name                    |
| Age                        | Numeric     | 18 - 70 years                    |
| Gender                     | Categorical | Male / Female                    |
| Location                   | Categorical | Houston, LA, Miami, Chicago, NYC |
| Subscription_Length_Months | Numeric     | 1 - 24 months                    |
| Monthly_Bill               | Numeric     | $30 - $100                       |
| Total_Usage_GB             | Numeric     | 50 - 500 GB                      |
| Churn                      | Binary      | 0 (Retained) / 1 (Churned)       |

---

## 🔬 Methodology

### 1️⃣ Exploratory Data Analysis

- Data quality assessment (no missing values, no duplicates)
- Distribution analysis (all features normally distributed)
- Correlation analysis (weak correlations found)

### 2️⃣ Data Preprocessing

- One-hot encoding for categorical variables
- MinMaxScaler for numerical features
- 70/30 train-test split

### 3️⃣ Feature Engineering

- Random Forest feature importance
- VIF analysis for multicollinearity (all < 5)
- PCA for dimensionality reduction

### 4️⃣ Models Evaluated

| Category          | Algorithms                                                |
| ----------------- | --------------------------------------------------------- |
| **Classical ML**  | Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes |
| **Ensemble**      | Random Forest, AdaBoost, Gradient Boosting, XGBoost       |
| **Deep Learning** | 5 Neural Network architectures with Keras                 |

### 5️⃣ Optimization

- GridSearchCV for hyperparameter tuning
- 5-fold Cross-validation
- Threshold optimization (0.1 - 1.0)

---

## 📈 Results

### Model Performance (Test Set)

| Model        | Accuracy | Precision | Recall | F1-Score |
| ------------ | -------- | --------- | ------ | -------- |
| **XGBoost**  | 50.1%    | 50.4%     | 52.7%  | 51.6%    |
| **Keras NN** | 50.1%    | 50.2%     | 98.1%  | 66.4%    |

### Key Findings

⚠️ **Important Insight:** The near-zero correlations between features and churn (~0.00) indicate that the provided features have **limited predictive power**. This explains why all models perform at approximately random chance level.

---

## 🖥️ Interactive Dashboard

A fully-featured Streamlit dashboard for model evaluation and data exploration.

### Features

- 📊 **Overview Tab:** Dataset statistics, churn distribution, feature statistics
- 🎯 **Model Performance:** Metrics, confusion matrix, ROC curve, feature importance
- 📈 **Visualizations:** Interactive charts and correlation heatmap
- 📋 **Data Explorer:** Filter data and download as CSV

### Screenshots

_Dashboard running at http://localhost:8502_

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "ML Project"

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Pre-compute Metrics (Optional)

```bash
python export_metrics.py
```

---

## 🛠️ Tech Stack

| Category             | Technologies              |
| -------------------- | ------------------------- |
| **Language**         | Python 3.11               |
| **Data Processing**  | Pandas, NumPy             |
| **Visualization**    | Matplotlib, Seaborn       |
| **Machine Learning** | Scikit-learn, XGBoost     |
| **Deep Learning**    | TensorFlow, Keras         |
| **Dashboard**        | Streamlit                 |
| **Development**      | Jupyter Notebook, VS Code |

---

## 📦 Dependencies

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
xgboost>=1.6
tensorflow>=2.6
streamlit>=1.10
matplotlib
seaborn
openpyxl
joblib
```

---

## 📄 Documentation

- **[Project Report](Customer_Churn_Prediction_Report.md)** - Comprehensive analysis report
- **[Jupyter Notebook](Customer%20Churn%20Prediction.ipynb)** - Step-by-step analysis
- **[Dashboard Code](app.py)** - Streamlit application

---

## 🔮 Future Improvements

1. **Data Enhancement**

   - Collect behavioral features (login frequency, support tickets)
   - Add customer lifetime value metrics
   - Include temporal patterns

2. **Model Improvements**

   - Implement SHAP for model interpretability
   - Try deep learning with embeddings
   - Ensemble multiple models

3. **Production Deployment**
   - Containerize with Docker
   - Deploy to cloud (Azure/AWS)
   - Implement monitoring and alerts

---

## 👨‍💻 Author

**Mohd Aafi**  
_12th January 2026_

---

## 📜 License

This project is for educational purposes as part of the Machine Learning Case Study.

---

## 🙏 References

- Scikit-learn documentation
- XGBoost documentation
- TensorFlow/Keras documentation
- Streamlit community
