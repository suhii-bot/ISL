import pandas as pd

# ---------------------------
# Load both datasets
# ---------------------------
numbers_df = pd.read_csv("data/isl_landmark_data.csv")
alphabets_df = pd.read_csv("data/isl_alphabets_data.csv")

# ---------------------------
# Merge datasets
# ---------------------------
merged_df = pd.concat([numbers_df, alphabets_df], ignore_index=True)

# ---------------------------
# Data Cleaning
# ---------------------------

# 1. Drop completely empty rows
merged_df.dropna(how='all', inplace=True)

# 2. Remove duplicate rows (if same features & label already exist)
merged_df.drop_duplicates(inplace=True)

# 3. Ensure label column exists and clean it
if 'label' not in merged_df.columns:
    raise KeyError("❌ The dataset must contain a 'label' column for gestures.")

# 4. Strip whitespace and make labels lowercase for consistency
merged_df['label'] = merged_df['label'].astype(str).str.strip().str.upper()

# 5. Remove rows with missing or invalid labels
merged_df = merged_df[merged_df['label'] != '']
merged_df.dropna(subset=['label'], inplace=True)

# ---------------------------
# Save the cleaned & merged file
# ---------------------------
merged_df.to_csv("data/isl_combined_data.csv", index=False)
print(f"✅ Merged & cleaned data saved to 'data/isl_combined_data.csv'")
print(f"📊 Final dataset shape: {merged_df.shape}")
print(f"🔍 Unique labels: {merged_df['label'].nunique()}")
