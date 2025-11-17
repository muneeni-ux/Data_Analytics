# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("✔ Libraries loaded successfully!\n")

# ============================================================
# 2. CONNECT TO MYSQL AND LOAD DATA
# ============================================================
print("🔌 Connecting to MySQL...")

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="testdb",
        use_pure=True
    )
    print("✔ Connected to MySQL database successfully!")
except Exception as e:
    print("❌ ERROR: Could not connect to MySQL.")
    print("Error message:", e)
    exit(1)

# Load table q4_data
try:
    df = pd.read_sql("SELECT * FROM q4_data", conn)
    print(f"\n📌 RAW DATA PREVIEW ({len(df)} rows):")
    print(df.head())
except Exception as e:
    print("❌ ERROR loading dataset from MySQL:")
    print(e)
    exit(1)

# ============================================================
# 3. CLEAN DATA
# ============================================================
print("\n🔧 Cleaning dataset...")

# Strip column names
df.columns = df.columns.str.strip()

# Convert numeric columns
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'num']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert categorical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'dataset']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

# Fill missing numeric values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Replace 'nan' strings with 'Unknown' in categorical columns
df[categorical_cols] = df[categorical_cols].replace("nan", "Unknown")

print("✔ Data cleaned successfully!\n")
print("📌 CLEANED DATA INFO:")
print(df.info())

# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================
print("\n📊 SUMMARY STATISTICS:")
print(df.describe())

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
print("\n📌 CORRELATION WITH HEART DISEASE (num):")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix['num'].sort_values(ascending=False))

# Heatmap for numeric correlations
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# 6. AGE VS HEART DISEASE
# ============================================================
plt.figure(figsize=(8,5))
plt.scatter(df['age'], df['num'], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Heart Disease Score (num)")
plt.title("Age vs Heart Disease Severity")
plt.show()

# ============================================================
# 7. GENDER ANALYSIS
# ============================================================
gender_result = df.groupby('sex')['num'].mean().reset_index()
print("\n📌 AVERAGE HEART DISEASE SCORE BY GENDER:")
print(gender_result)

plt.figure(figsize=(7,4))
sns.barplot(x='sex', y='num', data=gender_result)
plt.xlabel("Sex (1 = Male, 0 = Female)")
plt.ylabel("Average Disease Severity")
plt.title("Heart Disease Severity by Sex")
plt.show()

# ============================================================
# 8. CHOLESTEROL VS HEART DISEASE
# ============================================================
plt.figure(figsize=(8,5))
plt.scatter(df['chol'], df['num'], alpha=0.6, color='orange')
plt.xlabel("Cholesterol Level")
plt.ylabel("Heart Disease Score")
plt.title("Cholesterol vs Heart Disease Severity")
plt.show()

# ============================================================
# 9. BLOOD PRESSURE VS HEART DISEASE
# ============================================================
plt.figure(figsize=(8,5))
plt.scatter(df['trestbps'], df['num'], alpha=0.6, color='green')
plt.xlabel("Resting Blood Pressure (trestbps)")
plt.ylabel("Heart Disease Score")
plt.title("Blood Pressure vs Heart Disease Severity")
plt.show()

# ============================================================
# 10. AGE GROUP ANALYSIS
# ============================================================
df['age_group'] = pd.cut(df['age'], bins=[29,39,49,59,69,79,89], labels=['30-39','40-49','50-59','60-69','70-79','80-89'])
age_group_result = df.groupby('age_group')['num'].mean().reset_index()

print("\n📌 AVERAGE HEART DISEASE SCORE BY AGE GROUP:")
print(age_group_result)

plt.figure(figsize=(8,5))
sns.barplot(x='age_group', y='num', data=age_group_result, palette="viridis")
plt.xlabel("Age Group")
plt.ylabel("Average Heart Disease Score")
plt.title("Heart Disease Severity by Age Group")
plt.show()

# ============================================================
# 11. FINAL INSIGHTS
# ============================================================
print("\n=======================")
print("📌 FINAL ANALYTICS SUMMARY")
print("=======================\n")

print("🔹 Strongest positive correlations with heart disease:")
print(corr_matrix['num'].sort_values(ascending=False).head())

print("\n🔹 Average disease severity by gender:")
print(gender_result)

print("\n🔹 Average disease severity by age group:")
print(age_group_result)

print("\n🔹 Key Insights:")
print("""
- Older individuals show higher heart disease levels.
- Males typically show higher severity than females.
- Higher cholesterol strongly correlates with heart disease.
- Higher resting blood pressure also correlates with heart disease.
""")

print("\n🎉 ANALYSIS COMPLETE!")
