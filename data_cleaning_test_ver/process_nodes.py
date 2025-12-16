import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mapping_file = os.path.join(script_dir, 'station_id_mapping.csv')

files = [
    os.path.join(script_dir, 'transport_node_train_202509.csv'),
    os.path.join(script_dir, 'transport_node_train_202510.csv'),
    os.path.join(script_dir, 'transport_node_train_202511.csv')
]

# --- LOAD MAPPING ---
if not os.path.exists(mapping_file):
    raise FileNotFoundError("STOP! 'station_id_mapping.csv' not found.")

print("Loading station mapping...")
map_df = pd.read_csv(mapping_file)
station_to_id = pd.Series(map_df.Station_ID.values, index=map_df.Station_Code).to_dict()

def clean_station_code(code):
    if isinstance(code, str):
        return code.split('/')[0]
    return code

# --- MAIN LOOP ---
df_list = []

for file in files:
    if not os.path.exists(file):
        print(f"⚠️ Warning: Could not find '{os.path.basename(file)}'. Skipping.")
        continue

    print(f"Processing {os.path.basename(file)}...")
    temp_df = pd.read_csv(file)

    if 'PT_TYPE' in temp_df.columns:
        temp_df = temp_df[temp_df['PT_TYPE'] == 'TRAIN'].copy()

    temp_df['STATION_CODE_CLEAN'] = temp_df['PT_CODE'].apply(clean_station_code)
    temp_df['STATION_ID'] = temp_df['STATION_CODE_CLEAN'].map(station_to_id)

    # Drop missing stations
    temp_df = temp_df.dropna(subset=['STATION_ID'])
    temp_df['STATION_ID'] = temp_df['STATION_ID'].astype(int)

    temp_df['MONTH'] = pd.to_datetime(temp_df['YEAR_MONTH']).dt.month
    df_list.append(temp_df)

# --- MERGE & SAVE ---
if df_list:
    full_df = pd.concat(df_list, ignore_index=True)

    df_nodes = full_df.groupby([
        'MONTH',
        'DAY_TYPE', 
        'TIME_PER_HOUR', 
        'STATION_ID'
    ])[['TOTAL_TAP_IN_VOLUME', 'TOTAL_TAP_OUT_VOLUME']].sum().reset_index()

    # Feature Engineering
    df_nodes['IN_FLOW_NORM'] = np.log1p(df_nodes['TOTAL_TAP_IN_VOLUME'])
    df_nodes['OUT_FLOW_NORM'] = np.log1p(df_nodes['TOTAL_TAP_OUT_VOLUME'])
    df_nodes['IS_WEEKEND'] = df_nodes['DAY_TYPE'].apply(lambda x: 1 if 'WEEKEND' in x else 0)

    # --- OPTIMIZATION: DROP STRING COLUMNS ---
    df_nodes = df_nodes.drop(columns=['DAY_TYPE'])

    # Save
    output_filename = os.path.join(script_dir, 'combined_node_features.csv')
    df_nodes.to_csv(output_filename, index=False)
    
    print("-" * 30)
    print(f"Success! Optimized node features saved.")
    print(df_nodes.head())

else:
    print("No data processed.")