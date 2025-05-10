import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Config ===
data_dir = "building_datasets"
plot_dir = "prediction_plots"
os.makedirs(plot_dir, exist_ok=True)

performance = []

# === MAPE function ===
def mape(y_true, y_pred, threshold=0.1):
    mask = np.abs(y_true) > threshold
    if not np.any(mask):
        return 0.0  # If all actuals are too small, return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# === Loop through building files ===
for file in os.listdir(data_dir):
    if file.endswith("_dataset.csv"):
        building = file.replace("_dataset.csv", "")
        file_path = os.path.join(data_dir, file)
        print(f"Processing {building}...")

        # Load data
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Time features
        df['hour'] = df['Timestamp'].dt.hour
        df['dayofweek'] = df['Timestamp'].dt.dayofweek
        df['month'] = df['Timestamp'].dt.month

        # Train-validation split
        train_df = df[(df['Timestamp'] >= '2023-01-01') & (df['Timestamp'] < '2023-07-01')]
        val_df = df[(df['Timestamp'] >= '2023-07-01') & (df['Timestamp'] < '2024-01-01')]

        if train_df.empty or val_df.empty:
            print(f"Skipping {building} due to missing training or validation data.")
            continue

        # Feature selection
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        exclude_columns = ['Electricity_kWh', 'Area_m2', 'hour', 'dayofweek', 'month']
        features = [col for col in numeric_columns if col not in exclude_columns]
        features += ['hour', 'dayofweek', 'month']
        target = 'Electricity_kWh'

        X_train = train_df[features]
        y_train = train_df[target]
        X_val = val_df[features]
        y_val = val_df[target]

        # Train model
        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape_score = mape(y_val, y_pred)

        # Save plot
        plt.figure(figsize=(14, 5))
        plt.plot(val_df['Timestamp'], y_val.values, label='Actual', alpha=0.7)
        plt.plot(val_df['Timestamp'], y_pred, label='Predicted (XGBoost)', alpha=0.7)
        plt.title(f'XGBoost Regression: {building}')
        plt.xlabel('Timestamp')
        plt.ylabel('Electricity (kWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{building}_prediction.png")
        plt.savefig(plot_path)
        plt.close()

        # Store results
        performance.append({
            "Building": building,
            "MSE": round(mse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4),
            "MAPE (%)": round(mape_score, 2)
        })

# Save performance report
results_df = pd.DataFrame(performance)
results_df.to_csv("building_xgboost_performance.csv", index=False)
print("✅ All done! Metrics saved to 'building_xgboost_performance.csv'.")