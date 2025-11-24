# q4_crime_analysis_refined.py
# ================================================================
# Robust Crime Analysis pipeline for q5_data table (MySQL -> Pandas)
# - Flexible date parsing (avoids empty results)
# - Robust time parsing for TIME_OCC values like '1110', '1', '635'
# - Safe aliasing for AREA_NAME and Crm_Cd_Desc
# - Basic geospatial extraction if lat/lon present (optional)
# - Saves simple visualizations and CSV summaries
# ================================================================

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime

# -----------------------
# CONFIG - update as needed
# -----------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "admin",
    "database": "testdb",
    "use_pure": True,
    "connection_timeout": 10,
}

TABLE_NAME = "q5_data"
OUTPUT_DIR = "./q4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 1) Connect to MySQL
# -----------------------
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    print("✓ Database connected successfully")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    raise

# -----------------------
# 2) Load data
# -----------------------
try:
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    print(f"✓ Loaded table '{TABLE_NAME}' — rows: {len(df)}")
except Exception as e:
    print(f"✗ Failed to read table '{TABLE_NAME}': {e}")
    raise

# -----------------------
# 3) Normalize column names (strip spaces)
# -----------------------
df.columns = df.columns.str.strip()
# Keep a copy
orig_len = len(df)

# -----------------------
# 4) Flexible Date Parsing
#    - Date_Rptd and DATE_OCC may be stored as strings in many formats
#    - Use pd.to_datetime with errors="coerce" so invalid -> NaT
# -----------------------
# Try direct parse; this will handle most common formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
df["Date_Rptd"] = pd.to_datetime(df.get("Date_Rptd"), errors="coerce")
df["DATE_OCC"]   = pd.to_datetime(df.get("DATE_OCC"), errors="coerce")

print("✓ Date parsing summary:")
print(f"  Date_Rptd: {df['Date_Rptd'].notna().sum()}/{len(df)} parsed")
print(f"  DATE_OCC : {df['DATE_OCC'].notna().sum()}/{len(df)} parsed")

# If DATE_OCC mostly NaT, attempt some common fallback formats heuristics:
if df["DATE_OCC"].notna().sum() < max(1, int(len(df) * 0.2)):
    # Try parsing with dayfirst True (handles dd/mm/YYYY)
    fallback = pd.to_datetime(df.get("DATE_OCC"), errors="coerce", dayfirst=True)
    improved = fallback.notna().sum()
    if improved > df["DATE_OCC"].notna().sum():
        df["DATE_OCC"] = fallback
        print(f"  ✓ Improved DATE_OCC parsing using dayfirst. Now parsed: {improved}/{len(df)}")

# -----------------------
# 5) Robust TIME_OCC parsing
#    TIME_OCC can be '1110', '1', '635', sometimes non-digit — handle safely
# -----------------------
def fix_time(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Remove any non-digit characters (but keep digits)
    s_digits = re.sub(r"\D", "", s)
    if s_digits == "":
        return None
    # Ensure 4 digits (HHMM)
    s_digits = s_digits.zfill(4)
    try:
        t = pd.to_datetime(s_digits, format="%H%M", errors="coerce").time()
        return t
    except Exception:
        return None

df["TIME_OCC_parsed"] = df.get("TIME_OCC").apply(fix_time)

print(f"✓ TIME_OCC parsed (non-null): {df['TIME_OCC_parsed'].notna().sum()}/{len(df)}")

# -----------------------
# 6) Extract coordinates if LOCATION contains lat/lon pairs
#    If LOCATION is an address (as in your sample), this will remain NaN.
# -----------------------
def extract_coord(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)
    # Try to match explicit lat/lon patterns like "(34.05, -118.24)" or "34.05, -118.24"
    m = re.search(r"([-+]?\d{1,3}\.\d+)[, ]+\s*([-+]?\d{1,3}\.\d+)", s)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except:
            return (np.nan, np.nan)
    return (np.nan, np.nan)

coords = df.get("LOCATION", pd.Series([np.nan]*len(df))).apply(extract_coord)
df["Lat"] = coords.apply(lambda x: x[0] if isinstance(x, (tuple, list)) and len(x) >= 2 else np.nan)
df["Lon"] = coords.apply(lambda x: x[1] if isinstance(x, (tuple, list)) and len(x) >= 2 else np.nan)
print(f"✓ Coordinates found: Lat non-null {df['Lat'].notna().sum()}, Lon non-null {df['Lon'].notna().sum()}")

# -----------------------
# 7) Safe aliasing for important columns
# -----------------------
# Build a lowercase map to find similar columns flexibly
cols_lower = {c.lower(): c for c in df.columns}

def find_col(possible_names):
    # exact names first
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    # substring match
    for lname, orig in cols_lower.items():
        for name in possible_names:
            if name.lower() in lname:
                return orig
    return None

# AREA_NAME alias
area_col = find_col(["AREA_NAME", "area_name", "area", "neighborhood", "beat", "location"])
if area_col:
    df["AREA_NAME_alias"] = df[area_col].astype(str).fillna("Unknown")
else:
    df["AREA_NAME_alias"] = "Unknown"

# Crime description alias
crime_col = find_col(["Crm_Cd_Desc", "crm_cd_desc", "crm", "crime", "offense", "description"])
if crime_col:
    df["Crm_Cd_Desc_alias"] = df[crime_col].astype(str).fillna("Unknown")
else:
    df["Crm_Cd_Desc_alias"] = "Unknown"

# Ensure DATE_OCC exists (already parsed above)
if "DATE_OCC" not in df.columns or df["DATE_OCC"].isna().all():
    # try other candidate date columns
    fallback_date = find_col(["date_rptd", "date", "date_reported", "date_occ"])
    if fallback_date:
        df["DATE_OCC"] = pd.to_datetime(df[fallback_date], errors="coerce")
        print(f"✓ Mapped fallback '{fallback_date}' -> DATE_OCC parsed non-null: {df['DATE_OCC'].notna().sum()}")
    else:
        print("⚠ No usable date column found. Time-series will be empty.")

# -----------------------
# 8) Filter rows with a valid DATE_OCC but do not drop everything blindly
#    We'll keep rows that have a parsed DATE_OCC; if too many are missing warn user.
# -----------------------
total_rows = len(df)
valid_date_count = df["DATE_OCC"].notna().sum()
if valid_date_count == 0:
    print("✗ No valid DATE_OCC values found. Aborting time-series and geography steps.")
else:
    print(f"✓ Rows with valid DATE_OCC: {valid_date_count}/{total_rows}")

# Keep only rows with valid DATE_OCC for time-series & top-area analysis
df_time = df[df["DATE_OCC"].notna()].copy()

# -----------------------
# 9) Derive time features (Day / Week / Month / Year)
# -----------------------
if len(df_time) > 0:
    df_time["Day"] = df_time["DATE_OCC"].dt.date
    df_time["Week"] = df_time["DATE_OCC"].dt.isocalendar().week
    df_time["Month"] = df_time["DATE_OCC"].dt.to_period("M")
    df_time["Year"] = df_time["DATE_OCC"].dt.year

    # Monthly / Weekly / Daily groupings
    monthly_crime = df_time.groupby("Month").size().reset_index(name="Incidents")
    weekly_crime = df_time.groupby("Week").size().reset_index(name="Incidents")
    daily_crime  = df_time.groupby("Day").size().reset_index(name="Incidents")
    print("✓ Computed daily/weekly/monthly aggregates")
else:
    monthly_crime = pd.DataFrame(columns=["Month", "Incidents"])
    weekly_crime = pd.DataFrame(columns=["Week", "Incidents"])
    daily_crime = pd.DataFrame(columns=["Day", "Incidents"])

# -----------------------
# 10) Top Areas & Top Crime Types
# -----------------------
top_areas = df_time.groupby("AREA_NAME_alias").size().reset_index(name="Incidents").sort_values("Incidents", ascending=False)
top_crimes = df_time.groupby("Crm_Cd_Desc_alias").size().reset_index(name="Incidents").sort_values("Incidents", ascending=False)

print("\nTop Areas (top 10):")
print(top_areas.head(10).to_string(index=False))
print("\nTop Crime Types (top 10):")
print(top_crimes.head(10).to_string(index=False))

# -----------------------
# 11) Plots (save to OUTPUT_DIR)
# -----------------------
# Plot: Top 10 areas bar chart
if not top_areas.empty:
    plt.figure(figsize=(10,6))
    top_areas.head(10).plot.bar(x="AREA_NAME_alias", y="Incidents", legend=False, color="tab:green")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Crime Areas")
    plt.ylabel("Incidents")
    plt.tight_layout()
    path_top_areas = os.path.join(OUTPUT_DIR, "q4_top_areas.png")
    plt.savefig(path_top_areas)
    plt.close()
    print(f"✓ Saved Top Areas chart -> {path_top_areas}")
else:
    print("⚠ Top Areas plot skipped (no data).")

# Plot: Monthly trend
if not monthly_crime.empty:
    plt.figure(figsize=(10,6))
    # convert Period to string for plotting if necessary
    monthly_crime_plot = monthly_crime.copy()
    monthly_crime_plot["Month_str"] = monthly_crime_plot["Month"].astype(str)
    plt.plot(monthly_crime_plot["Month_str"], monthly_crime_plot["Incidents"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Crime Trend")
    plt.xlabel("Month")
    plt.ylabel("Incidents")
    plt.tight_layout()
    path_monthly = os.path.join(OUTPUT_DIR, "q4_monthly_trend.png")
    plt.savefig(path_monthly)
    plt.close()
    print(f"✓ Saved Monthly Trend chart -> {path_monthly}")
else:
    print("⚠ Monthly trend plot skipped (no data).")

# -----------------------
# 12) Export summaries (CSV) for quick review
# -----------------------
top_areas.head(100).to_csv(os.path.join(OUTPUT_DIR, "q4_top_areas.csv"), index=False)
top_crimes.head(100).to_csv(os.path.join(OUTPUT_DIR, "q4_top_crimes.csv"), index=False)
monthly_crime.to_csv(os.path.join(OUTPUT_DIR, "q4_monthly_crime.csv"), index=False)
daily_crime.to_csv(os.path.join(OUTPUT_DIR, "q4_daily_crime.csv"), index=False)
print(f"✓ Exported CSV summaries to {OUTPUT_DIR}")

# -----------------------
# 13) Brief Console Summary
# -----------------------
print("\n================ SUMMARY ================")
print(f"Input rows: {total_rows}")
print(f"Rows with parsed DATE_OCC: {valid_date_count}")
print(f"Unique areas detected: {df_time['AREA_NAME_alias'].nunique() if len(df_time)>0 else 0}")
print(f"Unique crime types detected: {df_time['Crm_Cd_Desc_alias'].nunique() if len(df_time)>0 else 0}")
print("Files saved in:", os.path.abspath(OUTPUT_DIR))
print("==========================================")

# Optionally display the first few cleaned rows for verification
print("\nSample cleaned rows:")
print(df_time.head(5).to_string(index=False))

# Done
conn.close()
