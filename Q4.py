# ================================================================
# Q4 – CRIME ANALYTICS (GEOGRAPHY + TIME SERIES) FROM MYSQL
# ================================================================

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ---------------------------------------------------------------
# 1. CONNECT TO MYSQL
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# 2. LOAD DATA FROM TABLE
# ---------------------------------------------------------------
try:
    df = pd.read_sql("SELECT * FROM q5_data", conn)   # <-- your table name
    print(f"✓ Crime data loaded: {len(df)} rows")
except Exception as e:
    print(f"✗ Failed loading: {e}")
    exit(1)


# ---------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ---------------------------------------------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ---------------------------------------------------------------
# 4. DATE & TIME CLEANING
# ---------------------------------------------------------------
def to_date(x):
    return pd.to_datetime(str(x), format="%m/%d/%Y", errors="coerce")

df["Date_Rptd"] = df["Date_Rptd"].apply(to_date)
df["DATE_OCC"]  = df["DATE_OCC"].apply(to_date)

# TIME_OCC is HHMM (e.g., 1430)
def fix_time(x):
    x = str(x).zfill(4)
    try:
        return pd.to_datetime(x, format="%H%M", errors="coerce").time()
    except:
        return None

df["TIME_OCC"] = df["TIME_OCC"].apply(fix_time)

# Remove entries where DATE_OCC couldn't be parsed
df = df.dropna(subset=["DATE_OCC"])

print("✓ Dates and times cleaned")


# ---------------------------------------------------------------
# 5. GEOSPATIAL PREP (Extract Lat/Lon if present)
# Example LOCATION: "(34.0522, -118.2437)"
# ---------------------------------------------------------------
def extract_coord(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    # Ensure we operate on a string
    s = str(text)
    match = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if len(match) >= 2:
        try:
            return float(match[0]), float(match[1])
        except Exception:
            return (np.nan, np.nan)
    return (np.nan, np.nan)

# Safely parse LOCATION -> Lat/Lon. Handle missing column or malformed entries.
if "LOCATION" in df.columns:
    parsed_coords = df["LOCATION"].apply(extract_coord)
    df["Lat"] = parsed_coords.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan)
    df["Lon"] = parsed_coords.apply(lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan)
    non_null = df["Lat"].notna().sum() + df["Lon"].notna().sum()
    print(f"✓ Coordinates extracted (if present). Non-null coord values: {non_null}")
else:
    df["Lat"] = np.nan
    df["Lon"] = np.nan
    print("⚠ 'LOCATION' column not found — Lat/Lon set to NaN")

# ---------------------------------------------------------------
# 5b. Normalize expected downstream column names (aliases)
# If the dataset uses different column names, try to detect them
# and create aliases `AREA_NAME`, `Crm_Cd_Desc`, and ensure `DATE_OCC` exists.
# ---------------------------------------------------------------
cols_lower = {c.lower(): c for c in df.columns}

def find_col_key(keywords):
    # Exact name match first
    for k in keywords:
        if k.lower() in cols_lower:
            return cols_lower[k.lower()]
    # Substring match
    for col_l, col in cols_lower.items():
        for k in keywords:
            if k in col_l:
                return col
    return None

# Area column
area_col = find_col_key(["area_name", "area", "neighborhood", "beat"]) or None
if area_col is not None:
    df["AREA_NAME"] = df[area_col].astype(str)
else:
    df["AREA_NAME"] = "Unknown"
    print("⚠ Could not detect an area column; alias 'AREA_NAME' set to 'Unknown'")

# Crime description column
crime_col = find_col_key(["crm_cd_desc", "crm", "crime", "offense", "description"]) or None
if crime_col is not None:
    df["Crm_Cd_Desc"] = df[crime_col].astype(str)
else:
    df["Crm_Cd_Desc"] = "Unknown"
    print("⚠ Could not detect a crime description column; alias 'Crm_Cd_Desc' set to 'Unknown'")

# Ensure DATE_OCC alias exists (if alternate date column present, keep existing DATE_OCC)
if "DATE_OCC" not in df.columns:
    date_fallback = find_col_key(["date_occ", "date_occur", "date_reported", "date", "occ_date"]) or None
    if date_fallback is not None:
        try:
            df["DATE_OCC"] = pd.to_datetime(df[date_fallback], errors="coerce")
            print(f"✓ Mapped fallback date column '{date_fallback}' to 'DATE_OCC'")
        except Exception:
            df["DATE_OCC"] = pd.NaT
            print(f"⚠ Failed to convert fallback date column '{date_fallback}' to datetime; 'DATE_OCC' set to NaT")
    else:
        df["DATE_OCC"] = pd.NaT
        print("⚠ No date column detected for 'DATE_OCC'; time-series steps may be empty")


# ---------------------------------------------------------------
# 6. TIME SERIES: DAILY, WEEKLY, MONTHLY
# ---------------------------------------------------------------
df["Month"] = df["DATE_OCC"].dt.to_period("M")
df["Week"]  = df["DATE_OCC"].dt.isocalendar().week
df["Day"]   = df["DATE_OCC"].dt.date

monthly_crime = (
    df.groupby("Month")
      .size()
      .reset_index(name="Incidents")
)

daily_crime = (
    df.groupby("Day")
      .size()
      .reset_index(name="Incidents")
)

weekly_crime = (
    df.groupby("Week")
      .size()
      .reset_index(name="Incidents")
)

print("✓ Time-series groups computed")


# ---------------------------------------------------------------
# 7. TOP AREAS (GEOGRAPHY)
# ---------------------------------------------------------------
top_areas = (
    df.groupby("AREA_NAME")
      .size()
      .reset_index(name="Incidents")
      .sort_values("Incidents", ascending=False)
)

print("\nTop Crime Areas:")
print(top_areas.head(10))


# ---------------------------------------------------------------
# 8. TOP CRIME TYPES
# ---------------------------------------------------------------
top_crimes = (
    df.groupby("Crm_Cd_Desc")
      .size()
      .reset_index(name="Incidents")
      .sort_values("Incidents", ascending=False)
)

print("\nTop Crime Types:")
print(top_crimes.head(10))


# ---------------------------------------------------------------
# 9. PLOTS — AREA CRIME & MONTHLY TREND
# ---------------------------------------------------------------

# ---- Plot 1: Top 10 crime areas ----
plt.figure(figsize=(10,6))
plt.bar(top_areas["AREA_NAME"].head(10), top_areas["Incidents"].head(10))
plt.xticks(rotation=45)
plt.title("Top 10 Crime Areas")
plt.ylabel("Incidents")
plt.tight_layout()
plt.savefig("q4_top_areas.png")
plt.show()

# ---- Plot 2: Monthly crime trend ----
plt.figure(figsize=(10,6))
plt.plot(monthly_crime["Month"].astype(str), monthly_crime["Incidents"])
plt.xticks(rotation=45)
plt.title("Monthly Crime Trend")
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.savefig("q4_monthly_trend.png")
plt.show()

print("\n✓ Crime visualizations saved:")
print(" - q4_top_areas.png")
print(" - q4_monthly_trend.png")


# ---------------------------------------------------------------
# 10. SUMMARY
# ---------------------------------------------------------------
print("\n================== SUMMARY ==================")
print("\nAreas with the highest crime:")
print(top_areas.head(10))

print("\nMost common crime types:")
print(top_crimes.head(10))

# Year-over-year trend:
df["Year"] = df["DATE_OCC"].dt.year
yearly_trend = df.groupby("Year").size().reset_index(name="Incidents")

print("\nYearly Crime Trend:")
print(yearly_trend)

print("\n====================================================")
print("✓ Q4 Crime Analysis Complete")
