## ⚡ TALTECH Building Energy Forecasting — Hackathon Solution

This repository contains my solution to the **TALTECH Campus Building Electricity Forecasting Hackathon**. The challenge was to **predict hourly electricity consumption for a new building** using:

* Historical hourly electricity data of several buildings (2023)
* Building area information
* Local hourly weather data (Tallinn)
* **Only 2 months** of data available for the "new" building during inference

---

## 🎯 Problem Statement

> Given the **first 2 months of electricity consumption** for a "new" building and full weather data, **predict hourly consumption for the next 10 months**.

Participants were evaluated on:

* 🔍 **Accuracy** (MAPE-based)
* ⚙️ **Computational efficiency**
* 🧠 **Elegance and clarity of presentation**

---

## 🧪 Solution Overview

* 📊 **Model**: XGBoost Regressor — chosen for its ability to handle tabular, structured, and nonlinear data efficiently

* 🏗️ **Features**:

  * Time-based: hour, weekday, month, sin/cos transforms
  * Weather-based: temperature, humidity, wind speed, dew point, visibility
  * Area-normalized consumption
  * Lag features: previous hour/day/week consumption
  * Engineered indices (e.g., discomfort index, hot/cold day flags)

* 🧼 **Preprocessing**: Unified, reproducible cleaning + feature generation for all buildings

* 🧠 **Prediction strategy**:

  * For each building: train model on first 2 or 6 months → predict next 10 or 6 months
  * Save predictions, visualizations, and metrics per building

---

## 📈 Model Performance

| Scenario                                  | Avg MAPE                       | Comment                        |
| ----------------------------------------- | ------------------------------ | ------------------------------ |
| **Train on 2 months → Predict 10 months** | \~**9.8%** (median much lower) | Good accuracy for limited data |
| **Train on 6 months → Predict 6 months**  | **<1%** in most buildings      | Excellent generalization       |

### 🏢 Per-Building Results

#### 🔹 Train 2 months → Predict 10 months

| Building            | MSE   | MAE  | R²     | MAPE (%) |
| ------------------- | ----- | ---- | ------ | -------- |
| MEK                 | 0.18  | 0.11 | 0.997  | 0.62     |
| U05, U04, U04B, GEO | 5.4   | 0.74 | 0.996  | 0.69     |
| ICT                 | 37.89 | 1.24 | 0.979  | 0.48     |
| D04                 | 0.43  | 0.26 | 0.987  | 4.99     |
| OBS                 | 13.04 | 2.82 | -0.95  | 194.44 ❗ |
| SOC                 | 0.61  | 0.37 | 0.9997 | 0.37     |
| TEG                 | 2.14  | 1.09 | 0.423  | 85.55 ❗  |
| LIB                 | 41.43 | 2.63 | 0.985  | 5.24     |
| S01                 | 0.08  | 0.16 | 0.9996 | 5.55     |
| U06, U06A, U05B     | 11    | 0.62 | 0.993  | 0.33     |

#### 🔹 Train 6 months → Predict 6 months

| Building            | MSE  | MAE  | R²     | MAPE (%) |
| ------------------- | ---- | ---- | ------ | -------- |
| MEK                 | 0.31 | 0.08 | 0.995  | 0.42     |
| U05, U04, U04B, GEO | 1.88 | 0.43 | 0.9986 | 0.46     |
| ICT                 | 4.73 | 0.4  | 0.9974 | 0.17     |
| D04                 | 0.01 | 0.02 | 0.9998 | 0.13     |
| OBS                 | 0    | 0.01 | 0.9999 | 0.49     |
| SOC                 | 0.3  | 0.26 | 0.9999 | 0.23     |
| TEG                 | 0    | 0    | 1.000  | 0.00     |
| LIB                 | 4.19 | 0.64 | 0.9982 | 1.76     |
| S01                 | 0.05 | 0.06 | 0.9997 | 0.41     |
| U06, U06A, U05B     | 0.68 | 0.2  | 0.9995 | 0.15     |

> ⚠️ Note: OBS and TEG show high MAPE in the 2-month scenario due to very low/near-zero actuals in some hours.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/dewanwadud1/taltech-energy-forecasting.git
cd taltech-energy-forecasting

# Install dependencies
pip install -r requirements.txt

# Run preprocessing (if not already done)
python pProcessing.py

# Run training + evaluation
python xGBoostRegression.py
```

Outputs:

* 📁 `prediction_plots/`: prediction vs. actual per building
* 📄 `building_xgboost_performance.csv`: all metrics in one file

---

## 📂 Folder Structure

```
.
├── building_datasets/           # Preprocessed per-building datasets
├── prediction_plots/            # Visual outputs per building
├── building_xgboost_performance.csv
├── pProcessing.py
├── xGBoostRegression.py
├── requirements.txt
└── README.md
```

---

## 🙌 Acknowledgements

* TALTECH Hackathon Committee
* Weather data: Tallinn METAR Archive
* XGBoost by DMLC
