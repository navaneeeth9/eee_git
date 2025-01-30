# Step 1: Install necessary libraries (if not installed already)
# !pip install plotly statsmodels scikit-learn pandas matplotlib seaborn

# Step 2: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 3: Load the CSV File (Manually enter the file name)
file_name = input("Enter the dataset file name (CSV format, e.g., 'climate_data.csv'): ").strip()
df = pd.read_csv(file_name)

# Step 4: Check for missing values and handle them
print("\nChecking for missing values...")
print(df.isnull().sum())
df.dropna(inplace=True)  # Drop rows with missing values

# Step 5: Convert Year column to numerical format (if not already done)
df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year

# Function to Predict Future Values
def predict_column(df, column_name):
    """Predict future values for a given column using linear regression."""
    X = df[['Year']]  # Features (Year)
    y = df[column_name]  # Target Variable

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict Future Values (Next 10 Years)
    future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 11).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    return future_predictions

# Step 6: Predict values for each column
predicted_temperature = predict_column(df, 'Global_Temperature_Anomaly_C')
predicted_co2 = predict_column(df, 'CO2_Concentration_ppm')
predicted_sea_level = predict_column(df, 'Sea_Level_Rise_mm')
predicted_deforestation = predict_column(df, 'Deforestation_Rate_Mha_per_year')

# Step 7: Create a new DataFrame for the future years and predictions
future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 11)

df_future = pd.DataFrame({
    'Year': future_years,
    'Global_Temperature_Anomaly_C': predicted_temperature,
    'CO2_Concentration_ppm': predicted_co2,
    'Sea_Level_Rise_mm': predicted_sea_level,
    'Deforestation_Rate_Mha_per_year': predicted_deforestation
})

# Step 8: Append the predicted data to the original DataFrame
df = pd.concat([df, df_future], ignore_index=True)

# Step 9: Save the modified DataFrame with predictions
output_file = "processed_climate_data_with_predictions.csv"
df.to_csv(output_file, index=False)

# Verify saved file by reading it back
df_check = pd.read_csv(output_file)
print("\nPredictions added! Here are the last 15 rows of the updated dataset:")
print(df_check.tail(15))  # Ensure last few rows include predictions

print(f"\nFile '{output_file}' has been saved successfully!")
