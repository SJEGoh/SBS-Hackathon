# Relevant libraries

import numpy as np
import pandas as pd

def clean_bus_data():
    """
    Clean and process bus passenger data.
    
    Returns:
        df_combined_5: DataFrame containing cleaned passenger load data for bus route 5
    """
    
    # Singapore Public Holidays for Apr-Jun 2025
    sg_holidays = [
        '2025-04-01',  # Hari Raya Puasa
        '2025-04-18',  # Good Friday
        '2025-05-01',  # Labour Day
        '2025-05-22',  # Vesak Day
        '2025-06-18',  # Hari Raya Haji
    ]
    
    def get_day_type(date):
        """Determine Day Type (WEEKDAY vs WEEKENDS/HOLIDAY)"""
        if date.day_name() in ['Saturday', 'Sunday']:
            return 'WEEKENDS/HOLIDAY'
        elif date.strftime('%Y-%m-%d') in sg_holidays:
            return 'WEEKENDS/HOLIDAY'
        else:
            return 'WEEKDAY'
    
    # ======================== PART I: PASSENGER LOAD ========================
    
    # Read transport volume data
    stops_files = ["data/transport_node_bus_202510.csv",
                   "data/transport_node_bus_202511.csv",
                   "data/transport_node_bus_202512.csv"]
    df_volume = pd.concat([pd.read_csv(f) for f in stops_files], ignore_index=True).rename(columns=str.strip).drop(["PT_TYPE"], axis=1)
    
    # Read bus stop codes for bus 5
    bus5_stops = pd.read_csv('data/bus5_stops.csv')
    
    # Filter df_volume for bus route 5
    df_volume_5 = df_volume[df_volume['PT_CODE'].isin(bus5_stops['PT_CODE'])]
    
    # Merge with bus stop data to get NO_OF_SERVICES and ID
    df_volume_5 = df_volume_5.merge(bus5_stops[['PT_CODE', 'NO_OF_SERVICES', 'ID']], on='PT_CODE', how='left')
    
    # Calculate the number of unique DAY_TYPEs per YEAR_MONTH
    df_volume_5['NUM_DAY_TYPES'] = df_volume_5.groupby('YEAR_MONTH')['DAY_TYPE'].transform('nunique')
    
    # Calculate NET_AVG_CHANGE = (TAP_IN - TAP_OUT) / (n * d)
    # where n = NO_OF_SERVICES, d = number of day types in that month
    df_volume_5['NET_AVG_CHANGE'] = (df_volume_5['TOTAL_TAP_IN_VOLUME'] - df_volume_5['TOTAL_TAP_OUT_VOLUME']) / (df_volume_5['NO_OF_SERVICES'] * df_volume_5['NUM_DAY_TYPES'])
    
    # Sort by YEAR_MONTH ascending, DAY_TYPE, TIME_PER_HOUR ascending, then by ID
    df_volume_5 = df_volume_5.sort_values(by=['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR', 'ID'], ascending=[True, True, True, True])
    
    # Drop the helper column
    df_volume_5 = df_volume_5.drop('NUM_DAY_TYPES', axis=1)
    
    # Read bus frequency data
    bus5_freq = pd.read_csv('data/bus5_freq.csv')
    
    # Merge frequency data with volume data
    df_volume_5 = df_volume_5.merge(bus5_freq[['TIME_PER_HOUR', 'DAY_TYPE', 'AVG_BUS_PER_HOUR']], on=['TIME_PER_HOUR', 'DAY_TYPE'], how='left')
    
    # Drop rows with NaN values in AVG_BUS_PER_HOUR
    df_volume_5 = df_volume_5.dropna(subset=['AVG_BUS_PER_HOUR'])
    
    # Calculate EST_LOAD using cumulative sum within each group, then divide by AVG_BUS_PER_HOUR
    df_volume_5['EST_LOAD'] = df_volume_5.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR'])['NET_AVG_CHANGE'].cumsum() / df_volume_5['AVG_BUS_PER_HOUR']
    
    # Create combined dataframe with mean and variance of EST_LOAD
    df_combined_5 = df_volume_5.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR']).agg(
        MEAN_EST_LOAD=('EST_LOAD', 'mean'),
        VARIANCE_EST_LOAD=('EST_LOAD', 'var')
    ).reset_index()

    # Convert all column names to uppercase
    df_combined_5.columns = df_combined_5.columns.str.upper()
    
    return df_combined_5

# Date, Bus, Model, Mileage start (hour), Mileage end (hour), Fuel efficiency
# Things to add (one hot): weekend, daily weather 

def fuel_efficiency_data():
    df = pd.read_excel("src/data/fuel_efficiency.xlsx")[["Date", "Bus", "Model", "Fuel Efficiency KML", "Mileage Start Time", "Mileage End Time", "Operational status", "Workshop activities"]]
    df = df.dropna()
    df = df[(df["Date"] >= "2025-04-01") & (df["Date"] <= "2025-06-30")]
    df["start_time"] = df["Mileage Start Time"].apply(lambda x: x.split()[1])
    df["end_time"] = df["Mileage End Time"].apply(lambda x: x.split()[1])
    df.drop(["Mileage Start Time", "Mileage End Time"], axis = 1, inplace = True)

    sg_holidays = [
        '2025-04-01',  # Hari Raya Puasa
        '2025-04-18',  # Good Friday
        '2025-05-01',  # Labour Day
        '2025-05-22',  # Vesak Day
        '2025-06-18',  # Hari Raya Haji
    ]

    sg_holidays = pd.to_datetime(sg_holidays)

    df["start_time"] = pd.to_datetime(df["start_time"].str.replace(r"\s*\+\d{2}(:?\d{2})?$", "", regex=True), utc=True, errors="coerce").dt.tz_localize(None).dt.round("h").dt.hour
    df["end_time"]   = pd.to_datetime(df["end_time"].str.replace(r"\s*\+\d{2}(:?\d{2})?$", "", regex=True),   utc=True, errors="coerce").dt.tz_localize(None).dt.round("h").dt.hour
    df["weekend"] = pd.to_numeric(df["Date"].dt.strftime("%w"))
    df["weekend/ph"] = np.where((df["weekend"] == 0) | (df["weekend"] == 6) | (df["weekend"].isin(sg_holidays)), 1, 0)

    # ======================== ADD WEATHER DATA ========================
    
    # Read and Combine Punggol CSVs
    punggol_files = ["data/DAILYDATA_Punggol_202504.csv", 
                     "data/DAILYDATA_Punggol_202505.csv",
                     "data/DAILYDATA_Punggol_202506.csv"]
    cols_to_keep_p = ["Year", "Month", "Day", "Daily Rainfall Total (mm)"]
    df_punggol = pd.concat([pd.read_csv(f)[cols_to_keep_p] for f in punggol_files], ignore_index=True)
    
    # Convert Day to datetime
    df_punggol['Date'] = pd.to_datetime(df_punggol['Day'].astype(str) + '-' + df_punggol['Month'].astype(str) + '-' + df_punggol['Year'].astype(str), format='%d-%m-%Y')
    
    # Read and Combine Seletar CSVs
    seletar_files = ["data/DAILYDATA_Seletar_202504.csv", 
                     "data/DAILYDATA_Seletar_202505.csv",
                     "data/DAILYDATA_Seletar_202506.csv"]
    cols_to_keep_s = ["Year", "Month", "Day", "Daily Rainfall Total (mm)", "Mean Temperature (°C)"]
    df_seletar = pd.concat([pd.read_csv(f)[cols_to_keep_s] for f in seletar_files], ignore_index=True)
    
    # Convert Day to datetime
    df_seletar['Date'] = pd.to_datetime(df_seletar['Day'].astype(str) + '-' + df_seletar['Month'].astype(str) + '-' + df_seletar['Year'].astype(str), format='%d-%m-%Y')
    
    # Keep only Daily Rainfall Total and Mean Temperature columns
    df_punggol = df_punggol[['Date', 'Daily Rainfall Total (mm)']].rename(columns={'Daily Rainfall Total (mm)': 'Punggol_Rainfall_mm'})
    df_seletar = df_seletar[['Date', 'Daily Rainfall Total (mm)', 'Mean Temperature (°C)']].rename(columns={'Daily Rainfall Total (mm)': 'Seletar_Rainfall_mm', 'Mean Temperature (°C)': 'Seletar_Temperature_C'})
    
    # Merge weather data with fuel efficiency data
    df = df.merge(df_punggol, on='Date', how='left')
    df = df.merge(df_seletar, on='Date', how='left')
    
    return df

# ok so now take start time, end time and calculate passenger load + variance across that period?

if __name__ == '__main__':
    # Call the function and get the dataframes
    '''
    data324, data329 = clean_bus_data()
    print("Data cleaned successfully!")
    print(f"data324 shape: {data324.shape}")
    print(f"data329 shape: {data329.shape}")
    print(data324.head())
    print(data324.columns)
    '''
    print(fuel_efficiency_data())



