# =======================================================
# Q3 – MEDIA / MOVIE ANALYTICS FROM MYSQL (IMDb Style)
# =======================================================

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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
# 2. LOAD DATA
# -------------------------------------------------------
try:
    df = pd.read_sql("SELECT * FROM q3_data", conn)
    print(f"✓ Data loaded successfully: {len(df)} rows")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    exit(1)


# -------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# -------------------------------------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")
print("✓ Columns cleaned")


# -------------------------------------------------------
# 4. TEXT CLEANING FUNCTIONS
# -------------------------------------------------------
def clean_text(x):
    if pd.isna(x): return ""
    x = str(x)
    x = re.sub(r'[\n\r\t]', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

# Apply cleaning to text columns
text_cols = ["Movie_Name", "Scraped_Name", "Director", "Writer", "Actor", "OtherInfo"]
for col in text_cols:
    df[col] = df[col].apply(clean_text)


# -------------------------------------------------------
# 5. CONVERT RATINGS TO NUMERIC
# -------------------------------------------------------
rating_cols = ["Rating", "DirectorsRating", "WritersRating", "TotalFollowers", "Revenue", "Budget"]

for col in rating_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .str.replace("$", "")
        .str.extract(r"(\d+\.?\d*)")  # extract numeric part
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("✓ Ratings converted to numeric")


# -------------------------------------------------------
# 6. EXTRACT YEAR FROM DATE
# -------------------------------------------------------
df["Year"] = df["Date"].astype(str).str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

print("✓ Year extracted from Date field")


# -------------------------------------------------------
# 7. BASIC ANALYTICS
# -------------------------------------------------------

# ---- Average rating by year ----
rating_by_year = (
    df.groupby("Year")["Rating"]
      .mean()
      .reset_index()
      .dropna()
)

# ---- Correlation between critics and audience ----
corr_matrix = df[["Rating", "DirectorsRating", "WritersRating"]].corr()


# -------------------------------------------------------
# 8. GENRE EXTRACTION (from Movie_Name or OtherInfo)
# -------------------------------------------------------

def extract_genre(text):
    """
    Attempts to extract genres from the movie info.
    Looks for common genres (extendable).
    """
    genres = [
        "Action", "Drama", "Comedy", "Thriller", "Horror",
        "Sci-Fi", "Fantasy", "Romance", "Adventure",
        "Documentary", "Animation", "Crime", "Mystery"
    ]
    found = []
    for g in genres:
        if g.lower() in str(text).lower():
            found.append(g)
    return found[0] if found else "Unknown"

df["Genre"] = df["OtherInfo"].apply(extract_genre)


# ---- Average rating by genre ----
top_genres = (
    df.groupby("Genre")["Rating"]
      .mean()
      .reset_index()
      .sort_values("Rating", ascending=False)
)

print("\nTop Genres by Average Rating:")
print(top_genres.head(10))


# -------------------------------------------------------
# 9. VISUALIZATIONS
# -------------------------------------------------------

# ---- Rating distribution ----
plt.figure(figsize=(8,5))
df["Rating"].hist(bins=20)
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("q3_rating_distribution.png")
plt.show()

# ---- Correlation heatmap (manual) ----
plt.figure(figsize=(6,4))
plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation: Ratings")
plt.tight_layout()
plt.savefig("q3_correlation_matrix.png")
plt.show()

# ---- Rating by Year trend ----
plt.figure(figsize=(10,5))
plt.plot(rating_by_year["Year"], rating_by_year["Rating"])
plt.title("Average Movie Rating by Release Year")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.grid(True)
plt.tight_layout()
plt.savefig("q3_rating_by_year.png")
plt.show()

print("\n✓ Visualizations saved:")
print(" - q3_rating_distribution.png")
print(" - q3_correlation_matrix.png")
print(" - q3_rating_by_year.png")

# -------------------------------------------------------
# 10. OUTPUT SUMMARY
# -------------------------------------------------------
print("\n===== SUMMARY =====")
print("Correlation Matrix:")
print(corr_matrix)

print("\nGenres Ranked by Rating:")
print(top_genres)

print("\nRating Trend by Year:")
print(rating_by_year)
