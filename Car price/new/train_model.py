# Filename: train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error # Added mean_absolute_error
import joblib
import numpy as np
# If you don't have scipy installed, you might need it for pearsonr:
# pip install scipy
from scipy.stats import pearsonr # Option 1 for Pearson r (more direct)

# --- 1. Load and Clean Data ---
print("Loading data...")
# Assumes Cleaned_Car_data.csv is in the same directory
try:
    df = pd.read_csv('Cleaned_Car_data.csv')
except FileNotFoundError:
    print("Error: 'Cleaned_Car_data.csv' not found in the current directory.")
    exit() # Stop execution if file not found

# Drop the unnecessary index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Basic cleaning
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Ensure Name is treated as string if it exists and needed, though not used as feature here
if 'Name' in df.columns:
    df['Name'] = df['Name'].astype(str)
df.dropna(subset=['Price', 'kms_driven', 'year', 'company', 'fuel_type'], inplace=True)
# Keep only numeric kms_driven and positive ones
df = df[pd.to_numeric(df['kms_driven'], errors='coerce').notna()]
df['kms_driven'] = df['kms_driven'].astype(int) # Ensure it's integer after filtering
df = df[df['kms_driven'] > 0] # Remove potential errors/placeholders
# Convert year to int
df['year'] = df['year'].astype(int)
# Remove rows where Price is non-positive
df = df[df['Price'] > 0]

df.reset_index(drop=True, inplace=True)

print(f"Data shape after cleaning: {df.shape}")
print("Sample data head:\n", df.head())

# --- 2. Feature Selection and Preparation ---
# Ensure only relevant columns are selected for features
X = df[['company', 'year', 'kms_driven', 'fuel_type']].copy() # Use .copy() to avoid SettingWithCopyWarning
y = df['Price'].copy()

# Optional: Log transform target variable if its distribution is skewed
# y = np.log1p(y) # Use log1p for robustness if Price can be 0, though we filtered > 0

print("\nFeatures (X) sample:\n", X.head())
print("\nTarget (y) sample:\n", y.head())

# --- 3. Train/Test Split ---
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- 4. Preprocessing and Model Pipeline ---
print("\nSetting up preprocessing and model pipeline...")
categorical_features = ['company', 'fuel_type']

# Ensure categorical features are treated as strings before OHE
X_train[categorical_features] = X_train[categorical_features].astype(str)
X_test[categorical_features] = X_test[categorical_features].astype(str)


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), # Added drop='first'
     categorical_features),
    remainder='passthrough' # Keeps numerical features as they are
)
lr_model = LinearRegression()
pipe = make_pipeline(column_trans, lr_model)

# --- 5. Train the Model ---
print("\nTraining the model...")
pipe.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate the Model ---
y_pred = pipe.predict(X_test)

# If you applied log transform to y, revert it for evaluation
# y_test = np.expm1(y_test)
# y_pred = np.expm1(y_pred)

# --- Calculate Metrics ---
print("\nModel Evaluation on Test Set:")

# R-squared (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)
print(f"  R-squared (R2): {r2:.4f}")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"  Mean Absolute Error (MAE): {mae:,.2f}") # Formatted for currency/price

# Pearson Correlation Coefficient (r)
# Option 1: Using scipy.stats.pearsonr
r_value, p_value = pearsonr(y_test, y_pred)
print(f"  Pearson Correlation Coefficient (r): {r_value:.4f}")
# Optional: Print p-value to check significance
# print(f"  Pearson Correlation p-value: {p_value:.4g}")

# Option 2: Using numpy.corrcoef (alternative)
# corr_matrix = np.corrcoef(y_test, y_pred)
# r_value_np = corr_matrix[0, 1]
# print(f"  Pearson Correlation Coefficient (r) [NumPy]: {r_value_np:.4f}")


# --- 7. Save the Model and Preprocessing Info ---
print("\nSaving the trained pipeline and lists...")
# Saves files to the same directory where the script is run
joblib.dump(pipe, 'car_price_model.pkl')

# Get unique values *after* cleaning and *before* splitting
# to represent all possible valid inputs the model was trained on conceptually
all_companies = sorted(list(df['company'].unique()))
all_fuel_types = sorted(list(df['fuel_type'].unique()))

joblib.dump(all_companies, 'companies.pkl')
joblib.dump(all_fuel_types, 'fuel_types.pkl')

print("Pipeline saved as car_price_model.pkl")
print(f"Saved {len(all_companies)} unique companies as companies.pkl")
print(f"Saved {len(all_fuel_types)} unique fuel types as fuel_types.pkl")

print("\nScript Finished.")