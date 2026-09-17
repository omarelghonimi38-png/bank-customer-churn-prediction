# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("data/Bank Customer Churn Prediction.csv")


# ==========================================
# 3. UNDERSTAND DATA
# ==========================================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA SHAPE =====")
print(df.shape)

print("\n===== DATA INFO =====")
df.info()

print("\n===== STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())


# ==========================================
# 4. EDA
# ==========================================

print("\n===== CHURN COUNTS =====")
print(df["churn"].value_counts())

print("\n===== CHURN PERCENTAGE =====")
print(df["churn"].value_counts(normalize=True))

print("\n===== CHURN BY GENDER =====")
print(pd.crosstab(df["gender"], df["churn"]))

print("\n===== CHURN BY COUNTRY =====")
print(pd.crosstab(df["country"], df["churn"]))

print("\n===== CHURN BY ACTIVE MEMBER =====")
print(pd.crosstab(df["active_member"], df["churn"]))


# ==========================================
# 5. FEATURES AND TARGET
# ==========================================

# X = المعلومات اللي الموديل هيتعلم منها
X = df.drop(columns=["customer_id", "churn"])

# y = الحاجة اللي عايزين نتوقعها
y = df["churn"]


# ==========================================
# 6. ENCODING
# ==========================================

# تحويل country و gender من Text إلى Numbers
X = pd.get_dummies(
    X,
    columns=["country", "gender"],
    drop_first=True,
    dtype=int
)

print("\n===== FEATURES AFTER ENCODING =====")
print(X.head())

print("\n===== FEATURE COLUMNS =====")
print(X.columns)


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN / TEST =====")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ==========================================
# 8. SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 9. EVALUATION TOOLS
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 10. LOGISTIC REGRESSION
# ==========================================

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()

# Training
log_model.fit(X_train_scaled, y_train)

# Prediction
log_pred = log_model.predict(X_test_scaled)

print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

print("Accuracy:")
print(accuracy_score(y_test, log_pred))

print("\nClassification Report:")
print(classification_report(y_test, log_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))


# ==========================================
# 11. DECISION TREE
# ==========================================

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42
)

# Training
dt_model.fit(X_train, y_train)

# Prediction
dt_pred = dt_model.predict(X_test)

print("\n================================")
print("DECISION TREE")
print("================================")

print("Accuracy:")
print(accuracy_score(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))


# ==========================================
# 12. RANDOM FOREST
# ==========================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
)


# Training
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

print("\n================================")
print("RANDOM FOREST")
print("================================")

print("Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 13. COMPARE MODELS
# ==========================================

log_accuracy = accuracy_score(y_test, log_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("Logistic Regression Accuracy:", log_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
# ==========================================
# 14. SAVE MODEL
# ==========================================

import joblib

joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel saved successfully!")
