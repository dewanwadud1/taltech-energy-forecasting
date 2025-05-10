import pandas as pd
import numpy as np
import os

def preprocess_all_buildings(file_path="Buildings_el.xlsx", output_folder="building_datasets"):
    os.makedirs(output_folder, exist_ok=True)

    # Load sheets
    electricity_df = pd.read_excel(file_path, sheet_name='Electricity kWh', skiprows=1)
    df_weather_raw = pd.read_excel(file_path, sheet_name='Weather archive', skiprows=2)
    area_df = pd.read_excel(file_path, sheet_name='Areas')

    # Weather data cleaning
    df_weather_raw = df_weather_raw.dropna(how='all', axis=1)
    df_weather_raw = df_weather_raw.loc[:, df_weather_raw.isnull().mean() < 0.9]
    df_weather_raw['WW'] = df_weather_raw['WW'].fillna('Clear')
    df_weather_raw = pd.concat([df_weather_raw, pd.get_dummies(df_weather_raw['WW'], prefix='WW')], axis=1)
    df_weather_raw['c'] = df_weather_raw['c'].fillna('Clear')
    df_weather_raw = pd.concat([df_weather_raw, pd.get_dummies(df_weather_raw['c'], prefix='Cloud')], axis=1)
    df_weather_raw.drop(columns=['WW', 'c'], inplace=True)

    for col in ['T', 'P0', 'P', 'U']:
        df_weather_raw[col] = df_weather_raw[col].interpolate(method='linear', limit_direction='both')

    df_weather_raw['VV'] = df_weather_raw['VV'].replace("10.0 and more", 10)
    df_weather_raw['VV'] = pd.to_numeric(df_weather_raw['VV'], errors='coerce')

    def compute_dewpoint(T, RH):
        a, b = 17.27, 237.7
        alpha = (a * T) / (b + T) + np.log(RH / 100.0)
        return (b * alpha) / (a - alpha)

    df_weather_raw['Td_computed'] = compute_dewpoint(df_weather_raw['T'], df_weather_raw['U'])
    df_weather_raw['DD'] = df_weather_raw['DD'].fillna('Unknown')
    df_weather_raw = pd.concat([df_weather_raw, pd.get_dummies(df_weather_raw['DD'], prefix='WindDir')], axis=1)
    df_weather_raw.drop(columns=['DD'], inplace=True)

    # Timestamp cleanup
    df_weather_raw.columns = df_weather_raw.columns.str.strip()
    df_weather_raw = df_weather_raw.rename(columns={'Local time in Tallinn': 'Timestamp'})
    df_weather_raw['Timestamp'] = df_weather_raw['Timestamp'].astype(str).str.strip()
    df_weather_raw['Timestamp'] = df_weather_raw['Timestamp'].str.replace(r'[\u200b\xa0\n\r\t]', '', regex=True)
    df_weather_raw['Timestamp'] = pd.to_datetime(df_weather_raw['Timestamp'], errors='coerce', dayfirst=True)
    df_weather_raw.dropna(subset=['Timestamp'], inplace=True)

    # Resample and filter
    weather_df = df_weather_raw.copy()
    weather_df['Timestamp'] = weather_df['Timestamp'].dt.floor('H')
    weather_df = weather_df.groupby('Timestamp').mean().reset_index()

    electricity_df = electricity_df.rename(columns={electricity_df.columns[0]: "Timestamp"})
    electricity_df['Timestamp'] = pd.to_datetime(electricity_df['Timestamp'])

    start_time = electricity_df['Timestamp'].min()
    end_time = electricity_df['Timestamp'].max()
    weather_df = weather_df[(weather_df['Timestamp'] >= start_time) & (weather_df['Timestamp'] <= end_time)]

    # Process each building
    for building in electricity_df.columns[1:]:
        power_df = electricity_df[['Timestamp', building]].copy()
        power_df.rename(columns={building: 'Electricity_kWh'}, inplace=True)

        try:
            area = area_df.loc[area_df['Buid_ID'] == building, 'Area [m2]'].values[0]
        except IndexError:
            area = np.nan

        power_df['Area_m2'] = area
        merged_df = pd.merge(power_df, weather_df, on='Timestamp', how='inner')

        # Feature Engineering
        merged_df['ProxyF'] = merged_df['Ff']**0.8 * (295 - merged_df['T'])
        merged_df['hour'] = merged_df['Timestamp'].dt.hour
        merged_df['dayofweek'] = merged_df['Timestamp'].dt.dayofweek
        merged_df['month'] = merged_df['Timestamp'].dt.month
        merged_df['weekofyear'] = merged_df['Timestamp'].dt.isocalendar().week
        merged_df['is_weekend'] = merged_df['dayofweek'].isin([5, 6])
        merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24)
        merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24)
        merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
        merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)
        merged_df['lag_1'] = merged_df['Electricity_kWh'].shift(1)
        merged_df['lag_24'] = merged_df['Electricity_kWh'].shift(24)
        merged_df['lag_168'] = merged_df['Electricity_kWh'].shift(168)
        merged_df['kWh_per_m2'] = merged_df['Electricity_kWh'] / merged_df['Area_m2']
        merged_df['hot_day'] = merged_df['T'] > 25
        merged_df['cold_day'] = merged_df['T'] < 5
        merged_df['temp_rolling_avg'] = merged_df['T'].rolling(window=24).mean()
        merged_df['temp_dev'] = merged_df['T'] - merged_df['temp_rolling_avg']
        merged_df['discomfort_index'] = 0.5 * (
            merged_df['T'] + 61.0 + ((merged_df['T'] - 68.0) * 1.2) + (merged_df['U'] * 0.094)
        )
        merged_df.dropna(inplace=True)

        # Save
        filename = building.replace("/", "_").replace("\\", "_") + "_dataset.csv"
        merged_df.to_csv(os.path.join(output_folder, filename), index=False)

    print(f"✅ Preprocessing complete. Datasets saved in '{output_folder}/'.")

# To run:
preprocess_all_buildings("Buildings_el.xlsx")
