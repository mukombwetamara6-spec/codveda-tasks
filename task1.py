import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# TASK 1: DATA CLEANING AND PREPROCESSING
# =====================================================================
print("\n--- Task 1: Starting Data Cleaning ---")

try:
    # FIXED: Replaced delim_whitespace=True with modern sep=r'\s+'
    df = pd.read_csv('4) house Prediction Data Set.csv', header=None, sep=r'\s+')
    print("Successfully loaded '4) house Prediction Data Set.csv'!")
except FileNotFoundError:
    print("Error: Could not find '4) house Prediction Data Set.csv'.")
    exit()

# Give the columns simple numeric labels (0, 1, 2, 3...) since there is no header row
df.columns = [str(i) for i in range(df.shape[1])]

print("\n--- Initial Dataset Overview ---")
print(df.info())
print("\nFirst 5 rows of data:")
print(df.head())
print("-" * 50)

# Identify and handle missing values
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Automatically handle missing values:
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print("\n[Fixed] Missing values have been handled.")

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Found {duplicate_count} duplicate rows.")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed successfully.")

print("\n--- Task 1 Completed Successfully ---\n")


# =====================================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("--- Task 2: Starting Exploratory Data Analysis ---")

# 1. Calculate Summary Statistics
print("\nStatistical Summary of Columns:")
print(df.describe())

print("\nMode of all columns:")
print(df.mode().iloc[0])
print("-" * 50)

# 2. Visualize Data Distributions
sns.set_theme(style="whitegrid")

# Select the very first numerical column to plot
target_col = numeric_cols[0]
print(f"Plotting distributions for column index: '{target_col}'")

plt.figure(figsize=(12, 5))

# Subplot 1: Histogram
plt.subplot(1, 2, 1)
sns.histplot(df[target_col], kde=True, color='royalblue')
plt.title(f'Distribution of Column {target_col}')

# Subplot 2: Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=df[target_col], color='coral')
plt.title(f'Column {target_col} Boxplot Outlier Check')

plt.tight_layout()
plt.show()

# 3. Find Correlations Between Numerical Features
print("\nCalculating Correlation Matrix...")
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

# Visualize correlation matrix using a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

print("\n--- Task 2 Completed Successfully ---")
# 4. Scatter Plot (Added to fully complete Task 2 requirements)
print("\nGenerating Scatter Plot...")
plt.figure(figsize=(7, 5))
# Using columns '5' and '13' as a great example of a relationship in this dataset
sns.scatterplot(x=df['5'], y=df['13'], color='darkviolet', alpha=0.7)
plt.title('Relationship Between Feature 5 and Feature 13')
plt.xlabel('Feature 5')
plt.ylabel('Feature 13')
plt.tight_layout()
plt.show()