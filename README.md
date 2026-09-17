# 🏦 Bank Customer Churn Prediction

A Machine Learning project that predicts whether a bank customer is likely to leave the bank (Churn) based on customer and banking information.

The project includes data preprocessing, exploratory data analysis, machine learning model training, model evaluation, and an interactive Streamlit application.

---

## 🎯 Project Objective

Customer churn is an important business problem for banks.

The goal of this project is to build a Machine Learning model that can identify customers who are at risk of leaving the bank.

This can help banks identify at-risk customers and support customer retention strategies.

---

## 📊 Dataset

The dataset contains 10,000 bank customers.

The main features include:

- Credit Score
- Country
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card
- Active Member
- Estimated Salary

### Target

The target variable is `churn`:

- `0` = Customer stays
- `1` = Customer churns

The dataset contains approximately:

- 79.63% non-churn customers
- 20.37% churn customers

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values
- Checked for duplicate records
- Removed `customer_id` from the model features
- Applied One-Hot Encoding to categorical variables
- Split the data into training and testing sets
- Applied StandardScaler for Logistic Regression
- Used `class_weight="balanced"` with Random Forest to improve churn detection

---

## 🤖 Machine Learning Models

Three classification models were tested:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.80% |
| Decision Tree | 78.30% |
| Random Forest | 84.75% |

### Random Forest Results

The balanced Random Forest achieved:

- Accuracy: 84.75%
- Churn Precision: 63%
- Churn Recall: 61%
- Churn F1-Score: 62%

Because the dataset is imbalanced, the model was evaluated using Precision, Recall, and F1-Score in addition to Accuracy.

---

## 🌲 Final Model

Random Forest is used in the final Streamlit application.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
)
```

---

## 🖥️ Streamlit Application

An interactive Streamlit application was created to make predictions using the trained model.

The user can enter:

- Credit Score
- Country
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card status
- Active Member status
- Estimated Salary

The application predicts whether the customer is:

- ✅ Likely to Stay
- ⚠️ Likely to Churn

The application also displays the predicted churn probability.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

```text
bank_customer_churn/
│
├── data/
│   └── Bank Customer Churn Prediction.csv
│
├── main.py
├── app.py
├── random_forest_model.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 👤 Author

**Omar**

Data Engineering & Machine Learning Learning Journey