# ================================================
# Q2 – CUSTOMER CHURN PREDICTION (BINARY CLASSIFICATION)
# LOAD DATA FROM MYSQL → CLEAN → TRAIN MODEL → ANALYZE
# ================================================

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# 1. CONNECT TO MYSQL
# -------------------------------------------------------
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="testdb",
        use_pure=True,
        connection_timeout=10,
        autocommit=True
    )
    print("✓ Database connected successfully")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    exit(1)

# -------------------------------------------------------
# 2. LOAD DATA FROM MYSQL TABLE
# -------------------------------------------------------
try:
    df = pd.read_sql("SELECT * FROM q2_data", conn)
    print(f"✓ Data loaded successfully: {len(df)} rows")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    exit(1)

# -------------------------------------------------------
# 3. CLEANING COLUMN NAMES
# -------------------------------------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")
print("\nCleaned Columns:", df.columns.tolist())

# -------------------------------------------------------
# 4. DATA CLEANING
# -------------------------------------------------------

# Remove ID fields
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True, errors="ignore")
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

# Convert TotalCharges from VARCHAR → numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges with median
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Tenure: fill missing with 0
df["tenure"].fillna(0, inplace=True)

# -------------------------------------------------------
# 5. FEATURE ENGINEERING
# -------------------------------------------------------

# Avg monthly spend
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

# Tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
)

# -------------------------------------------------------
# 6. ENCODING CATEGORICAL VARIABLES
# -------------------------------------------------------

label = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label.fit_transform(df[col].astype(str))

# Encode TenureGroup after it's created
if "TenureGroup" in df.columns:
    df["TenureGroup"] = label.fit_transform(df["TenureGroup"].astype(str))

print("\n✓ Encoding completed")

# -------------------------------------------------------
# 7. SPLIT DATA INTO TRAIN/TEST
# -------------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✓ Train/Test ready")

# -------------------------------------------------------
# 8. TRAIN MODEL (Random Forest)
# -------------------------------------------------------
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

print("\n✓ Model training complete")

# -------------------------------------------------------
# 9. PREDICTIONS & EVALUATION
# -------------------------------------------------------
pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
print("\nConfusion Matrix:\n", cm)

# -------------------------------------------------------
# 10. FEATURE IMPORTANCE (Top Factors Causing Churn)
# -------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop 10 Churn Indicators:")
print(importance.head(10))

# -------------------------------------------------------
# 11. VISUALIZE FEATURE IMPORTANCE
# -------------------------------------------------------
plt.figure(figsize=(10,6))
importance.head(10).plot(kind='bar')
plt.title("Top 10 Churn Indicators")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("churn_feature_importance.png")
plt.show()
print("\n✓ Feature importance chart saved as 'churn_feature_importance.png'")

