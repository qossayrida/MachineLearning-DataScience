import pandas as pd


# Load the dataset
file_path = '/mnt/data/cars.csv'
df = pd.read_csv(file_path)

# Step 1: Clean the data
# Check for missing values and handle them
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing numeric values with median
df.dropna(inplace=True)  # Drop rows with any remaining missing values in categorical features

