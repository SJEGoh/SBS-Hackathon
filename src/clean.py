# Relevant libraries

import numpy as np
import pandas as pd

def clean_bus_data():
    """
    Clean and process bus passenger data with weather and humidity information.
    
    Returns:
        tuple: (data324, data329) - DataFrames containing cleaned data for bus routes 324 and 329
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
    
    # ======================== PART I: RAINFALL, TEMPERATURE ========================
    
    # Read and Combine Punggol CSVs
    punggol_files = ["data/DAILYDATA_Punggol_202504.csv", 
                     "data/DAILYDATA_Punggol_202505.csv",
                     "data/DAILYDATA_Punggol_202506.csv"]
    cols_to_keep_p = ["Year", "Month", "Day", "Daily Rainfall Total (mm)"]
    df_punggol = pd.concat([pd.read_csv(f)[cols_to_keep_p] for f in punggol_files], ignore_index=True)
    
    # Convert Day to datetime
    df_punggol['Date'] = pd.to_datetime(df_punggol['Day'].astype(str) + '-' + df_punggol['Month'].astype(str) + '-' + df_punggol['Year'].astype(str), format='%d-%m-%Y')
    df_punggol['Day Type'] = df_punggol['Date'].apply(get_day_type)
    
    # Group by Month and Day Type, calculate mean and variance of rainfall
    data_punggol = df_punggol.groupby(['Month', 'Day Type'])['Daily Rainfall Total (mm)'].agg(
        Mean_Rainfall_Punggol='mean',
        Variance_Rainfall_Punggol='var'
    ).reset_index()
    
    # Read and Combine Seletar CSVs
    seletar_files = ["data/DAILYDATA_Seletar_202504.csv", 
                     "data/DAILYDATA_Seletar_202505.csv",
                     "data/DAILYDATA_Seletar_202506.csv"]
    cols_to_keep_s = ["Year", "Month", "Day", "Daily Rainfall Total (mm)", "Mean Temperature (°C)"]
    df_seletar = pd.concat([pd.read_csv(f)[cols_to_keep_s] for f in seletar_files], ignore_index=True)
    
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
    
    # ======================== PART II: RELATIVE HUMIDITY ========================
    
    # Read CSV
    humid_df = pd.read_csv('data/RelativeHumidityMonthlyMean.csv')
    
    # Filter for 2025-04, 2025-05, and 2025-06
    data_humidity = humid_df[humid_df['month'].str.contains('2025-(04|05|06)', regex=True)].copy()
    
    # Rename the mean-rh column to RELATIVE_HUMIDITY
    if 'mean-rh' in data_humidity.columns:
        data_humidity = data_humidity.rename(columns={'mean-rh': 'RELATIVE_HUMIDITY'})
    elif 'mean_rh' in data_humidity.columns:
        data_humidity = data_humidity.rename(columns={'mean_rh': 'RELATIVE_HUMIDITY'})
    
    # ======================== PART III: PASSENGER LOAD ========================
    
    # Read transport volume data
    stops_files = ["data/transport_node_bus_202510.csv",
                   "data/transport_node_bus_202511.csv",
                   "data/transport_node_bus_202512.csv"]
    df_volume = pd.concat([pd.read_csv(f) for f in stops_files], ignore_index=True).rename(columns=str.strip).drop(["PT_TYPE"], axis=1)
    
    # Read bus stop codes for each route
    bus324_stops = pd.read_csv('data/bus324_stops.csv')
    bus329_stops = pd.read_csv('data/bus329_stops.csv')
    
    # Filter df_volume for each bus route
    df_volume_324 = df_volume[df_volume['PT_CODE'].isin(bus324_stops['PT_CODE'])]
    df_volume_329 = df_volume[df_volume['PT_CODE'].isin(bus329_stops['PT_CODE'])]
    
    # Merge with bus stop data to get NO_OF_SERVICES and ID
    df_volume_324 = df_volume_324.merge(bus324_stops[['PT_CODE', 'NO_OF_SERVICES', 'ID']], on='PT_CODE', how='left')
    df_volume_329 = df_volume_329.merge(bus329_stops[['PT_CODE', 'NO_OF_SERVICES', 'ID']], on='PT_CODE', how='left')
    
    # Calculate the number of unique DAY_TYPEs per YEAR_MONTH for each bus route
    df_volume_324['NUM_DAY_TYPES'] = df_volume_324.groupby('YEAR_MONTH')['DAY_TYPE'].transform('nunique')
    df_volume_329['NUM_DAY_TYPES'] = df_volume_329.groupby('YEAR_MONTH')['DAY_TYPE'].transform('nunique')
    
    # Calculate NET_AVG_CHANGE = (TAP_IN - TAP_OUT) / (n * d)
    # where n = NO_OF_SERVICES, d = number of day types in that month
    df_volume_324['NET_AVG_CHANGE'] = (df_volume_324['TOTAL_TAP_IN_VOLUME'] - df_volume_324['TOTAL_TAP_OUT_VOLUME']) / (df_volume_324['NO_OF_SERVICES'] * df_volume_324['NUM_DAY_TYPES'])
    df_volume_329['NET_AVG_CHANGE'] = (df_volume_329['TOTAL_TAP_IN_VOLUME'] - df_volume_329['TOTAL_TAP_OUT_VOLUME']) / (df_volume_329['NO_OF_SERVICES'] * df_volume_329['NUM_DAY_TYPES'])
    
    # Sort by YEAR_MONTH ascending, DAY_TYPE, TIME_PER_HOUR ascending, then by ID
    df_volume_324 = df_volume_324.sort_values(by=['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR', 'ID'], ascending=[True, True, True, True])
    df_volume_329 = df_volume_329.sort_values(by=['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR', 'ID'], ascending=[True, True, True, True])
    
    # Drop the helper column
    df_volume_324 = df_volume_324.drop('NUM_DAY_TYPES', axis=1)
    df_volume_329 = df_volume_329.drop('NUM_DAY_TYPES', axis=1)
    
    # Read bus frequency data
    bus324_freq = pd.read_csv('data/bus324_freq.csv')
    bus329_freq = pd.read_csv('data/bus329_freq.csv')
    
    # Merge frequency data with volume data
    df_volume_324 = df_volume_324.merge(bus324_freq[['TIME_PER_HOUR', 'DAY_TYPE', 'AVG_BUS_PER_HOUR']], on=['TIME_PER_HOUR', 'DAY_TYPE'], how='left')
    df_volume_329 = df_volume_329.merge(bus329_freq[['TIME_PER_HOUR', 'DAY_TYPE', 'AVG_BUS_PER_HOUR']], on=['TIME_PER_HOUR', 'DAY_TYPE'], how='left')
    
    # Drop rows with NaN values in AVG_BUS_PER_HOUR
    df_volume_324 = df_volume_324.dropna(subset=['AVG_BUS_PER_HOUR'])
    df_volume_329 = df_volume_329.dropna(subset=['AVG_BUS_PER_HOUR'])
    
    # Calculate EST_LOAD using cumulative sum within each group, then divide by AVG_BUS_PER_HOUR
    df_volume_324['EST_LOAD'] = df_volume_324.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR'])['NET_AVG_CHANGE'].cumsum() / df_volume_324['AVG_BUS_PER_HOUR']
    df_volume_329['EST_LOAD'] = df_volume_329.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR'])['NET_AVG_CHANGE'].cumsum() / df_volume_329['AVG_BUS_PER_HOUR']
    
    # Create combined dataframes with mean and variance of EST_LOAD
    df_combined_324 = df_volume_324.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR']).agg(
        MEAN_EST_LOAD=('EST_LOAD', 'mean'),
        VARIANCE_EST_LOAD=('EST_LOAD', 'var')
    ).reset_index()
    
    df_combined_329 = df_volume_329.groupby(['YEAR_MONTH', 'DAY_TYPE', 'TIME_PER_HOUR']).agg(
        MEAN_EST_LOAD=('EST_LOAD', 'mean'),
        VARIANCE_EST_LOAD=('EST_LOAD', 'var')
    ).reset_index()
    
    # ======================== MERGE WEATHER AND HUMIDITY DATA ========================
    
    # Add YEAR_MONTH to data_punggol and data_seletar
    data_punggol['YEAR_MONTH'] = '2025-' + data_punggol['Month'].astype(str).str.zfill(2)
    data_seletar['YEAR_MONTH'] = '2025-' + data_seletar['Month'].astype(str).str.zfill(2)
    
    # Rename 'Day Type' to 'DAY_TYPE' for consistency
    data_punggol = data_punggol.rename(columns={'Day Type': 'DAY_TYPE'})
    data_seletar = data_seletar.rename(columns={'Day Type': 'DAY_TYPE'})
    
    # Extract YEAR_MONTH from the 'month' column in data_humidity
    data_humidity['YEAR_MONTH'] = data_humidity['month'].str.extract(r'(\d{4}-\d{2})')[0]
    
    # Map Apr-Jun weather data to Oct-Dec (Apr→Oct, May→Nov, Jun→Dec)
    month_mapping = {'2025-04': '2025-10', '2025-05': '2025-11', '2025-06': '2025-12'}
    
    data_punggol_mapped = data_punggol.copy()
    data_punggol_mapped['YEAR_MONTH'] = data_punggol_mapped['YEAR_MONTH'].map(month_mapping)
    
    data_seletar_mapped = data_seletar.copy()
    data_seletar_mapped['YEAR_MONTH'] = data_seletar_mapped['YEAR_MONTH'].map(month_mapping)
    
    data_humidity_mapped = data_humidity.copy()
    data_humidity_mapped['YEAR_MONTH'] = data_humidity_mapped['YEAR_MONTH'].map(month_mapping)
    
    # Merge weather data with combined dataframes using mapped data
    df_combined_324 = df_combined_324.merge(
        data_punggol_mapped[['YEAR_MONTH', 'DAY_TYPE', 'Mean_Rainfall_Punggol', 'Variance_Rainfall_Punggol']], 
        on=['YEAR_MONTH', 'DAY_TYPE'], 
        how='left'
    )
    df_combined_324 = df_combined_324.merge(
        data_seletar_mapped[['YEAR_MONTH', 'DAY_TYPE', 'Mean_Rainfall_Seletar', 'Variance_Rainfall_Seletar', 'Mean_Temperature_Seletar']], 
        on=['YEAR_MONTH', 'DAY_TYPE'], 
        how='left'
    )
    df_combined_324 = df_combined_324.merge(
        data_humidity_mapped[['YEAR_MONTH', 'RELATIVE_HUMIDITY']], 
        on='YEAR_MONTH', 
        how='left'
    )
    
    df_combined_329 = df_combined_329.merge(
        data_punggol_mapped[['YEAR_MONTH', 'DAY_TYPE', 'Mean_Rainfall_Punggol', 'Variance_Rainfall_Punggol']], 
        on=['YEAR_MONTH', 'DAY_TYPE'], 
        how='left'
    )
    df_combined_329 = df_combined_329.merge(
        data_seletar_mapped[['YEAR_MONTH', 'DAY_TYPE', 'Mean_Rainfall_Seletar', 'Variance_Rainfall_Seletar', 'Mean_Temperature_Seletar']], 
        on=['YEAR_MONTH', 'DAY_TYPE'], 
        how='left'
    )
    df_combined_329 = df_combined_329.merge(
        data_humidity_mapped[['YEAR_MONTH', 'RELATIVE_HUMIDITY']], 
        on='YEAR_MONTH', 
        how='left'
    )
    
    # Convert all column names to uppercase
    df_combined_324.columns = df_combined_324.columns.str.upper()
    df_combined_329.columns = df_combined_329.columns.str.upper()
    
    return df_combined_324, df_combined_329


if __name__ == '__main__':
    # Call the function and get the dataframes
    data324, data329 = clean_bus_data()
    print("Data cleaned successfully!")
    print(f"data324 shape: {data324.shape}")
    print(f"data329 shape: {data329.shape}")
    print(data324.head())
    print(data324.columns)


