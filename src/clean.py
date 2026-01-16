# Relevant libraries

import numpy as np
import pandas as pd
import json

"""PART I: RAINFALL, TEMPERATURE"""

# Read and Combine Punggol CSVs

punggol_files = ["data/DAILYDATA_Punggol_202510.csv", 
                 "data/DAILYDATA_Punggol_202511.csv",
                 "data/DAILYDATA_Punggol_202512.csv"]
cols_to_keep_p = ["Year",
                "Month",
                "Day",
                "Daily Rainfall Total (mm)"]
df_punggol = pd.concat([pd.read_csv(f)[cols_to_keep_p] for f in punggol_files], ignore_index = True)

# Singapore Public Holidays for Oct-Dec 2025
sg_holidays = [
    '2025-11-01',  # Deepavali
    '2025-12-25',  # Christmas Day
]

# Convert Day to datetime
df_punggol['Date'] = pd.to_datetime(df_punggol['Day'].astype(str) + '-' + df_punggol['Month'].astype(str) + '-' + df_punggol['Year'].astype(str), format='%d-%m-%Y')

# Determine Day Type (WEEKDAY vs WEEKENDS/HOLIDAY)
def get_day_type(date):
    if date.day_name() in ['Saturday', 'Sunday']:
        return 'WEEKENDS/HOLIDAY'
    elif date.strftime('%Y-%m-%d') in sg_holidays:
        return 'WEEKENDS/HOLIDAY'
    else:
        return 'WEEKDAY'

df_punggol['Day Type'] = df_punggol['Date'].apply(get_day_type)

# Group by Month and Day Type, calculate mean and variance of rainfall
data_punggol = df_punggol.groupby(['Month', 'Day Type'])['Daily Rainfall Total (mm)'].agg(
    Mean_Rainfall_Punggol='mean',
    Variance_Rainfall_Punggol='var'
).reset_index()

# Read and Combine Seletar CSVs

seletar_files = ["data/DAILYDATA_Seletar_202510.csv", 
                 "data/DAILYDATA_Seletar_202511.csv",
                 "data/DAILYDATA_Seletar_202512.csv"]
cols_to_keep_s = ["Year",
                "Month",
                "Day",
                "Daily Rainfall Total (mm)",
                "Mean Temperature (°C)"]
df_seletar = pd.concat([pd.read_csv(f)[cols_to_keep_s] for f in seletar_files], ignore_index = True)

# Convert Day to datetime
df_seletar['Date'] = pd.to_datetime(df_seletar['Day'].astype(str) + '-' + df_seletar['Month'].astype(str) + '-' + df_seletar['Year'].astype(str), format='%d-%m-%Y')

df_seletar['Day Type'] = df_seletar['Date'].apply(get_day_type)

# Group by Month and Day Type, calculate mean and variance of rainfall and temperature
data_seletar = df_seletar.groupby(['Month', 'Day Type']).agg(
    Mean_Rainfall_Seletar=('Daily Rainfall Total (mm)', 'mean'),
    Variance_Rainfall_Seletar=('Daily Rainfall Total (mm)', 'var'),
    Mean_Temperature_Seletar=('Mean Temperature (°C)', 'mean'),
    Variance_Temperature_Seletar=('Mean Temperature (°C)', 'var')
).reset_index()

"""PART II: RELATIVE HUMIDITY"""

# Read CSV

humid_df = pd.read_csv('data/RelativeHumidityMonthlyMean.csv')

# Filter for 2025-10, 2025-11, and 2025-12
humid_df_filtered = humid_df[df['month'].str.contains('2025-(10|11|12)', regex=True)]

"""PART III: PASSENGER LOAD"""

stops_files = ["data/transport_node_bus_202510.csv",
               "data/transport_node_bus_202511.csv",
               "data/transport_node_bus_202512.csv"]

df_volume = pd.concat([pd.read_csv(f) for f in stops_files], ignore_index = True).drop(['PT_TYPE'])

# Read bus stop codes for each route
bus324_stops = pd.read_csv('data/bus324_stops.csv')
bus329_stops = pd.read_csv('data/bus329_stops.csv')

# Filter df_volume for each bus route
df_volume_324 = df_volume[df_volume['PT_CODE'].isin(bus324_stops['PT_CODE'])]
df_volume_329 = df_volume[df_volume['PT_CODE'].isin(bus329_stops['PT_CODE'])]

# Load Bus Routes JSON
with open("bus_routes.json", "r") as f:
    temp = json.load(f)

# Convert to DataFrame
df_routes = pd.DataFrame(temp)