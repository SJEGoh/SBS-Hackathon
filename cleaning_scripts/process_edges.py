import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# 1. Determine where this script is located (inside 'cleaning_scripts')
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define the Data Directory (Go up one level '..', then into 'data')
# This logic works regardless of your computer's specific path
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

# Input Files (Located in ../data/)
files = [
    os.path.join(data_dir, 'origin_destination_train_202509.csv'),
    os.path.join(data_dir, 'origin_destination_train_202510.csv'), 
    os.path.join(data_dir, 'origin_destination_train_202511.csv')
]

def clean_station_code(code):
    if isinstance(code, str):
        return code.split('/')[0]
    return code

# --- MAIN LOOP ---
df_list = []
print(f"📂 Looking for data in: {data_dir}")
print("Starting EDGE processing (Origin-Destination Flow)...")

for file in files:
    if not os.path.exists(file):
        print(f"Error: Could not find '{os.path.basename(file)}'. Skipping.")
        continue
        
    print(f"Processing {os.path.basename(file)}...")
    temp_df = pd.read_csv(file)
    
    # 1. Clean Station Codes
    temp_df['ORIGIN_CLEAN'] = temp_df['ORIGIN_PT_CODE'].apply(clean_station_code)
    temp_df['DEST_CLEAN'] = temp_df['DESTINATION_PT_CODE'].apply(clean_station_code)
    
    # 2. Filter for TRAIN
    if 'PT_TYPE' in temp_df.columns:
        temp_df = temp_df[temp_df['PT_TYPE'] == 'TRAIN'].copy()
    
    # 3. Extract Month
    temp_df['MONTH'] = pd.to_datetime(temp_df['YEAR_MONTH']).dt.month
    df_list.append(temp_df)

# --- MERGE & MAP STATIONS ---
if df_list:
    full_df = pd.concat(df_list, ignore_index=True)

    print("Generating unique Station IDs for the Graph...")
    
    # Create the Master List of Stations (Nodes)
    all_stations = pd.unique(full_df[['ORIGIN_CLEAN', 'DEST_CLEAN']].values.ravel('K'))
    station_to_id = {station: i for i, station in enumerate(all_stations)}
    
    # Map Names to IDs
    full_df['ORIGIN_ID'] = full_df['ORIGIN_CLEAN'].map(station_to_id)
    full_df['DEST_ID'] = full_df['DEST_CLEAN'].map(station_to_id)
    
    # SAVE THE MAPPING to the 'data' folder
    mapping_df = pd.DataFrame(list(station_to_id.items()), columns=['Station_Code', 'Station_ID'])
    mapping_path = os.path.join(data_dir, 'station_id_mapping.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved global station mapping to '{mapping_path}'")

    # --- AGGREGATION ---
    df_flow = full_df.groupby([
        'MONTH',            
        'DAY_TYPE', 
        'TIME_PER_HOUR', 
        'ORIGIN_ID',       
        'DEST_ID'          
    ])['TOTAL_TRIPS'].sum().reset_index()

    # Feature Engineering
    df_flow['IS_WEEKEND'] = df_flow['DAY_TYPE'].apply(lambda x: 1 if 'WEEKEND' in x else 0)
    df_flow['NORMALIZED_FLOW'] = np.log1p(df_flow['TOTAL_TRIPS'])

    # Drop string columns to save memory
    df_flow = df_flow.drop(columns=['DAY_TYPE'])

    # Save Output to 'data' folder
    output_filename = os.path.join(data_dir, 'combined_train_flow_mapped.csv')
    df_flow.to_csv(output_filename, index=False)
    
    print("-" * 30)
    print(f"✅ Success! Edge data saved to: {os.path.basename(output_filename)}")
    print(df_flow.head())

else:
    print("No data loaded.")