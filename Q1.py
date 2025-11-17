import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Connect to MySQL
# ----------------------------
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

try:
    df = pd.read_sql("SELECT * FROM q1_data", conn)
    print(f"✓ Data loaded: {len(df)} rows")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    exit(1)

# ----------------------------
# 2. Data Cleaning
# ----------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")

df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%m/%d/%Y", errors='coerce')
df['Ship_Date']  = pd.to_datetime(df['Ship_Date'],  format="%m/%d/%Y", errors='coerce')

# Remove rows where date failed conversion
df = df.dropna(subset=['Order_Date'])

df['Postal_Code'] = df['Postal_Code'].fillna(0)
df['Region']      = df['Region'].fillna("Unknown")
df['Category']    = df['Category'].str.title().fillna("Misc")

# ----------------------------
# 3. Top Products
# ----------------------------
top_products = (
    df.groupby('Product_Name')
      .size()
      .reset_index(name='transactions')
      .sort_values('transactions', ascending=False)
      .head(10)
)

print("\nTop Products:")
print(top_products)

# ----------------------------
# 4. Top Regions
# ----------------------------
top_regions = (
    df.groupby('Region')
      .size()
      .reset_index(name='transactions')
      .sort_values('transactions', ascending=False)
)

print("\nTop Regions:")
print(top_regions)

# ----------------------------
# 5. Monthly Trends
# ----------------------------
df['Month'] = df['Order_Date'].dt.to_period('M')

monthly_trends = (
    df.groupby('Month')
      .size()
      .reset_index(name='transactions')
)

print("\nMonthly Trend:")
print(monthly_trends)

# ----------------------------
# 6. Plot Trends
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(monthly_trends['Month'].astype(str), monthly_trends['transactions'])
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("trend_plot.png")
print("\n✓ Plot saved as 'trend_plot.png'")
plt.show()

# ----------------------------
# 7. To be Promoted
# ----------------------------
regional_strength = (
    df.groupby(['Product_Name', 'Region'])
      .size()
      .reset_index(name='transactions')
)

strong_products = (
    regional_strength.groupby('Product_Name')['transactions']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

strong_products
print("\nProducts to be Promoted:")
print(strong_products)

# Sales high/low
monthly_trends.sort_values('transactions', ascending=False).head()
monthly_trends.sort_values('transactions').head()
print("\nMonths with Highest Sales:")
print(monthly_trends.sort_values('transactions', ascending=False).head())
print("\nMonths with Lowest Sales:")
print(monthly_trends.sort_values('transactions').head())