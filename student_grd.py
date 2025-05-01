# Data Preprocessing Implementation for Students Grading Dataset
# --------------------------------------------------------------
# This script implements all five steps of data preprocessing:
# 1. Data Cleaning
# 2. Data Integration 
# 3. Data Reduction
# 4. Data Transformation
# 5. Data Discretization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 0: Loading and Exploring the Dataset
# -----------------------------------------
print("Step 0: Loading and Exploring the Dataset")
print("-----------------------------------------")

# Load the Students Grading Dataset
df = pd.read_csv("Students_Grading.csv")

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Analyzing the distribution of numerical features
plt.figure(figsize=(15, 12))  # Adjust figure size for better layout
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(num_cols)
nrows = int(np.ceil(num_plots / 3))  # Calculate number of rows dynamically
for i, col in enumerate(num_cols):
    plt.subplot(nrows, 3, i+1)  # Use calculated nrows
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_distributions.png')

# Analyzing categorical features
plt.figure(figsize=(12, 8))
categorical_cols = df.select_dtypes(include=['object']).columns
# Calculate the number of rows and columns needed for subplots
num_categorical_cols = len(categorical_cols)
ncols = 2  # You can adjust this to change the number of columns
nrows = int(np.ceil(num_categorical_cols / ncols))

for i, col in enumerate(categorical_cols):
    plt.subplot(nrows, ncols, i+1)  # Use calculated nrows and ncols
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_distributions.png')

# Step 1: Data Cleaning
# --------------------
print("\nStep 1: Data Cleaning")
print("--------------------")

# Create a copy for cleaning
df_cleaned = df.copy()

# 1.1 Handling Missing Values
print("1.1 Handling Missing Values")
if df_cleaned.isnull().sum().sum() > 0:
    # Handling missing values for numerical columns using mean imputation
    num_imputer = SimpleImputer(strategy='mean')
    numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    df_cleaned[numerical_cols] = num_imputer.fit_transform(df_cleaned[numerical_cols])
    
    # Handling missing values for categorical columns using most frequent value
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
    
    print("Missing values after imputation:")
    print(df_cleaned.isnull().sum())
else:
    print("No missing values found in the dataset.")

# 1.2 Handling Duplicates
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows.")
else:
    print("No duplicate rows found.")

# 1.3 Handling Outliers using IQR method
print("\n1.3 Handling Outliers")
def detect_and_handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        print(f"Detected {outlier_count} outliers in {column}")
        # Cap outliers instead of removing them
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        print(f"Capped outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
    else:
        print(f"No outliers detected in {column}")
    
    return df

# Apply outlier detection and handling to numerical columns
for col in numerical_cols:
    if col not in ['Grade']:  # We don't want to modify the target variable
        df_cleaned = detect_and_handle_outliers(df_cleaned, col)

# Step 2: Data Integration
# -----------------------
print("\nStep 2: Data Integration")
print("-----------------------")

# For demonstration, let's simulate adding new features derived from existing ones
# This could represent combining data from different sources in a real scenario

print("Creating new features from existing data (data integration):")

# Create a new feature: Study Efficiency (ratio of Grade to Study Hours)
# Assuming the column name is 'StudyHours' instead of 'Study Hours'
df_cleaned['Study_Efficiency'] = df_cleaned['Grade'] / df_cleaned['StudyHours']  
print("Created 'Study_Efficiency' = Grade / StudyHours")

# Create interaction features
# Assuming the column name is 'StudyHours' instead of 'Study Hours'
df_cleaned['Class_Study_Interaction'] = df_cleaned['Class Attendance'] * df_cleaned['StudyHours']  
print("Created 'Class_Study_Interaction' = Class Attendance * StudyHours")

# Binary indicator for high attendance
df_cleaned['High_Attendance'] = (df_cleaned['Class Attendance'] > df_cleaned['Class Attendance'].median()).astype(int)
print("Created 'High_Attendance' - binary indicator for above median attendance")

print("\nDataset shape after integration:", df_cleaned.shape)
print("New features created:", ['Study_Efficiency', 'Class_Study_Interaction', 'High_Attendance'])

# Step 3: Data Reduction
# --------------------
print("\nStep 3: Data Reduction")
print("--------------------")

# Separate features and target
X = df_cleaned.drop('Grade', axis=1)
y = df_cleaned['Grade']

# Handle categorical features before reduction
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Create preprocessing pipeline for categorical features
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Apply transformations
if len(categorical_cols) > 0:
    X_cat = pd.DataFrame(categorical_transformer.fit_transform(X[categorical_cols]))
    X_cat.columns = categorical_transformer.get_feature_names_out(categorical_cols)
    X_num = X[numerical_cols].reset_index(drop=True)
    X_transformed = pd.concat([X_num, X_cat], axis=1)
else:
    X_transformed = X[numerical_cols].copy()

print(f"Shape after handling categorical features: {X_transformed.shape}")

# 3.1 Principal Component Analysis (PCA)
print("\n3.1 Applying PCA")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original number of features: {X_transformed.shape[1]}")
print(f"Number of features after PCA: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Visualize cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.savefig('pca_explained_variance.png')

# 3.2 Feature Selection using SelectKBest
print("\n3.2 Feature Selection using SelectKBest")
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X_transformed, y)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = X_transformed.columns[selected_indices]

print(f"Selected features: {selected_features.tolist()}")
print(f"Feature scores:")
for feature, score in zip(X_transformed.columns, selector.scores_):
    print(f"{feature}: {score:.4f}")

# Step 4: Data Transformation
# -------------------------
print("\nStep 4: Data Transformation")
print("-------------------------")

# 4.1 Min-Max Normalization
print("4.1 Applying Min-Max Normalization")
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X_transformed)

# Convert back to DataFrame for clarity
X_normalized_df = pd.DataFrame(X_normalized, columns=X_transformed.columns)

print("First few rows after normalization:")
print(X_normalized_df.head())
print("\nStatistics after normalization:")
print(X_normalized_df.describe())

# Visualize distribution before and after normalization for a sample feature
if len(numerical_cols) > 0:
    sample_col = numerical_cols[0]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(X[sample_col], kde=True)
    plt.title(f'Before Normalization: {sample_col}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(X_normalized_df[sample_col], kde=True)
    plt.title(f'After Normalization: {sample_col}')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png')

# Step 5: Data Discretization
# -------------------------
print("\nStep 5: Data Discretization")
print("-------------------------")

# 5.1 Equal-width binning
print("5.1 Applying Equal-width Discretization")

# Select a numerical feature for discretization
if 'Study Hours' in df_cleaned.columns:
    feature_to_discretize = 'Study Hours'
elif len(numerical_cols) > 0:
    feature_to_discretize = numerical_cols[0]
else:
    feature_to_discretize = None

if feature_to_discretize:
    n_bins = 4
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    # Get the feature data and reshape for the discretizer
    feature_data = df_cleaned[feature_to_discretize].values.reshape(-1, 1)
    
    # Apply discretization
    discretized_data = discretizer.fit_transform(feature_data)
    
    # Add discretized feature to the dataframe
    df_cleaned[f'{feature_to_discretize}_Discretized'] = discretized_data
    
    # Map bin indices to meaningful categories
    bin_edges = discretizer.bin_edges_[0]
    bin_labels = [f'Bin {i+1} ({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})' for i in range(n_bins)]
    
    print(f"Discretized {feature_to_discretize} into {n_bins} bins:")
    for i, label in enumerate(bin_labels):
        count = (df_cleaned[f'{feature_to_discretize}_Discretized'] == i).sum()
        print(f"{label}: {count} instances")
    
    # Visualize original vs discretized values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_cleaned[feature_to_discretize], kde=True)
    plt.title(f'Original: {feature_to_discretize}')
    # here it is
    plt.subplot(1, 2, 2)
    sns.countplot(x=df_cleaned[f'{feature_to_discretize}_Discretized'])
    plt.title(f'Discretized: {feature_to_discretize}')
    plt.xlabel('Bin')
    plt.xticks(range(n_bins), bin_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('discretization_comparison.png')
else:
    print("No suitable numerical feature found for discretization.")

# Final Dataset Summary
# ------------------
print("\nFinal Preprocessed Dataset Summary")
print("----------------------------------")
print(f"Original dataset shape: {df.shape}")
print(f"Final preprocessed dataset shape: {df_cleaned.shape}")
print("\nFeatures added:")
print("- Study_Efficiency")
print("- Class_Study_Interaction")
print("- High_Attendance")
if feature_to_discretize:
    print(f"- {feature_to_discretize}_Discretized")

# Save the preprocessed dataset
df_cleaned.to_csv('preprocessed_students_data.csv', index=False)
print("\nPreprocessed dataset saved as 'preprocessed_students_data.csv'")

# PCA-transformed data
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Grade'] = y.values
pca_df.to_csv('pca_transformed_data.csv', index=False)
print("PCA-transformed dataset saved as 'pca_transformed_data.csv'")

print("\nData preprocessing complete!")
