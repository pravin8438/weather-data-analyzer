# Weather Data Analyzer - Indian Developer Style
# Developed by: Gajendiran K

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load the weather data from CSV file
df = pd.read_csv("data/weather_data.csv")

# Convert 'Date' column to datetime and format for display
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df["Display_Date"] = df["Date"].dt.strftime("%d-%m-%Y")  # Indian-style format

# Fill missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Round numerical values
df["Temperature"] = df["Temperature"].round(0).astype(int)
df["Humidity"] = df["Humidity"].round(0).astype(int)
df["Rainfall"] = df["Rainfall"].round(0).astype(int)

# Create additional time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days

# Print first 10 rows with Indian-style date format
print("\nFirst 10 Records (Formatted):\n")
print(df[["Display_Date", "Temperature", "Humidity", "Rainfall"]].head(10).to_string(index=False))

# Show summary statistics
print("\nWeather Summary Statistics:\n")
print(df[["Temperature", "Humidity", "Rainfall"]].describe())

# --- Visualizations ---
sns.set_style("whitegrid")

# 1. Line Chart - Temperature Trend
plt.figure(figsize=(10, 4))
sns.lineplot(data=df, x="Date", y="Temperature", color="orange")
plt.title("Temperature Trend Over Time")
plt.ylabel("Temperature (°C)")
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# 2. Bar Graph - Yearly Rainfall
plt.figure(figsize=(8, 4))
df.groupby("Year")["Rainfall"].sum().plot(kind="bar", color="skyblue")
plt.title("Year-wise Rainfall")
plt.ylabel("Rainfall (mm)")
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# 3. Scatter Plot - Temperature vs Humidity
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Humidity", y="Temperature", color="green", alpha=0.6)
plt.title("Temperature vs Humidity")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# --- Linear Regression Forecast ---
X = df[["Days_Since_Start"]]
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nTemperature Forecast RMSE: {rmse:.2f} °C")

# Predict next 30 days
future_days = pd.DataFrame({
    "Days_Since_Start": np.arange(df["Days_Since_Start"].max() + 1, df["Days_Since_Start"].max() + 31)
})
future_predictions = model.predict(future_days).round(0)

# Plot forecast
plt.figure(figsize=(10, 4))
plt.scatter(X, y, alpha=0.3, label="Actual")
plt.plot(X, model.predict(X), color="orange", label="Trend Line")
plt.plot(future_days["Days_Since_Start"], future_predictions, color="red", linestyle="--", label="Forecast")
plt.title("Future Temperature Forecast (Next 30 Days)")
plt.xlabel("Days Since First Record")
plt.ylabel("Predicted Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()
