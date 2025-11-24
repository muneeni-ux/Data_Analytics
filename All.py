# ============================================================
# JUPYTER NOTEBOOK: Multi-Project Data Analysis
# Datasets: Superstore Sales, Telco Churn, IMDb Movies,
# Crime Data, Public Health (Heart Disease)
# ============================================================

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid')
print("✔ Libraries loaded successfully!\n")

# ============================================================
# 1. CONNECT TO MYSQL
# ============================================================
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="testdb",
        use_pure=True
    )
    print("✔ Connected to MySQL database successfully!\n")
except Exception as e:
    print(f"❌ ERROR: Could not connect to MySQL: {e}")

# ============================================================
# 2. Q1: SALES PERFORMANCE AND TREND ANALYSIS
# ============================================================
print("===== Q1: Sales Performance =====")

# Load data
df_q1 = pd.read_sql("SELECT * FROM q1_data", conn)

# Clean data
df_q1.columns = df_q1.columns.str.strip()
df_q1['Order_Date'] = pd.to_datetime(df_q1['Order_Date'], format="%m/%d/%Y", errors='coerce')
df_q1['Ship_Date'] = pd.to_datetime(df_q1['Ship_Date'], format="%m/%d/%Y", errors='coerce')
df_q1 = df_q1.dropna(subset=['Order_Date'])
df_q1['Region'] = df_q1['Region'].fillna('Unknown')
df_q1['Category'] = df_q1['Category'].str.title().fillna('Misc')

# Top products
top_products = df_q1.groupby('Product_Name').size().reset_index(name='transactions')\
                    .sort_values('transactions', ascending=False).head(10)
print("\nTop Products:")
print(top_products)

# Top regions
top_regions = df_q1.groupby('Region').size().reset_index(name='transactions')\
                   .sort_values('transactions', ascending=False)
print("\nTop Regions:")
print(top_regions)

# Monthly trends
df_q1['Month'] = df_q1['Order_Date'].dt.to_period('M')
monthly_trends = df_q1.groupby('Month').size().reset_index(name='transactions')
print("\nMonthly Trends:")
print(monthly_trends.head())

# Plot trend
plt.figure(figsize=(10,5))
plt.plot(monthly_trends['Month'].astype(str), monthly_trends['transactions'], marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.show()

# ============================================================
# 3. Q2: CUSTOMER CHURN PREDICTION ANALYSIS
# ============================================================
print("===== Q2: Customer Churn Analysis =====")

# Load data
df_q2 = pd.read_sql("SELECT * FROM q2_data", conn)

# Clean data
df_q2.columns = df_q2.columns.str.strip()
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
df_q2[num_cols] = df_q2[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
            'PaymentMethod','Churn']
df_q2[cat_cols] = df_q2[cat_cols].fillna('Unknown').astype(str)

# Churn counts
churn_counts = df_q2['Churn'].value_counts()
print("\nChurn Distribution:")
print(churn_counts)

# Plot churn
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df_q2)
plt.title("Customer Churn Distribution")
plt.show()

# Correlation
numeric_corr = df_q2[num_cols + ['Churn']].copy()
numeric_corr['Churn'] = numeric_corr['Churn'].map({'Yes':1, 'No':0})
print("\nCorrelation with Churn:")
print(numeric_corr.corr()['Churn'].sort_values(ascending=False))

# ============================================================
# 4. Q3: MOVIE RATINGS & GENRE ANALYSIS
# ============================================================
print("===== Q3: Movie Ratings Analysis =====")

# Load data
df_q3 = pd.read_sql("SELECT * FROM q3_data", conn)
df_q3.columns = df_q3.columns.str.strip()

# Convert numeric columns
num_cols_q3 = ['Rating', 'DirectorsRating', 'WritersRating', 'TotalFollowers', 'Revenue']
for col in num_cols_q3:
    df_q3[col] = pd.to_numeric(df_q3[col], errors='coerce')

# Top-rated movies
top_movies = df_q3[['Movie_Name','Rating']].sort_values('Rating', ascending=False).head(10)
print("\nTop Rated Movies:")
print(top_movies)

# Plot critics vs audience
plt.figure(figsize=(7,5))
plt.scatter(df_q3['DirectorsRating'], df_q3['Rating'], alpha=0.6)
plt.xlabel("Director's Rating")
plt.ylabel("Audience Rating")
plt.title("Director vs Audience Rating")
plt.show()

# ============================================================
# 5. Q4: PUBLIC HEALTH / HEART DISEASE ANALYSIS
# ============================================================
print("===== Q4: Heart Disease Analysis =====")

df_q4 = pd.read_sql("SELECT * FROM q4_data", conn)
df_q4.columns = df_q4.columns.str.strip()

num_cols_q4 = ['age','trestbps','chol','thalch','oldpeak','num']
cat_cols_q4 = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
df_q4[num_cols_q4] = df_q4[num_cols_q4].apply(pd.to_numeric, errors='coerce').fillna(0)
df_q4[cat_cols_q4] = df_q4[cat_cols_q4].fillna('Unknown').astype(str)

# Summary stats
print("\nSummary Statistics:")
print(df_q4.describe())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_q4[num_cols_q4].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Age vs Heart Disease
plt.figure(figsize=(7,5))
plt.scatter(df_q4['age'], df_q4['num'])
plt.xlabel("Age")
plt.ylabel("Heart Disease Score")
plt.title("Age vs Heart Disease Severity")
plt.show()

# ============================================================
# 6. Q5: CRIME DATA ANALYSIS
# ============================================================
print("===== Q5: Crime Data Analysis =====")

df_q5 = pd.read_sql("SELECT * FROM q5_data", conn)
df_q5.columns = df_q5.columns.str.strip()

# Convert dates
df_q5['DATE_OCC'] = pd.to_datetime(df_q5['DATE_OCC'], errors='coerce')
df_q5 = df_q5.dropna(subset=['DATE_OCC'])

# Crimes by AREA_NAME
area_counts = df_q5.groupby('AREA_NAME').size().reset_index(name='crime_count')\
                    .sort_values('crime_count', ascending=False)
print("\nTop Crime Areas:")
print(area_counts.head(10))

# Daily trend
df_q5['Date'] = df_q5['DATE_OCC'].dt.date
daily_trends = df_q5.groupby('Date').size().reset_index(name='crime_count')

plt.figure(figsize=(12,5))
plt.plot(daily_trends['Date'], daily_trends['crime_count'])
plt.title("Daily Crime Trend")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.show()
