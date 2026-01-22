from clean import clean_bus_data, fuel_efficiency_data, merge_data
from ML_model import ML_model
from kalman import Kalman
import numpy as np
import sqlite3
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def main():
    # define good values as not within 2 days of workshop day
    # split data into train + actual
    # actual store into a sql database

    data = merge_data()
    data = data[data["Operational status"] == 5]
    train_split = int(len(data) * 0.8)
    train_data = data[:train_split]
    test_data = data[train_split:]

    good_train = train_data[train_data["Workshop activities"] != "W"]
    print(good_train.columns)
    # this model predict fuel efficiency
    model = ML_model()
    print(good_train.columns)
    model.fit(good_train["Fuel Efficiency KML"], good_train[["weekend/ph", "Punggol_Rainfall_mm",
                                                            "Seletar_Rainfall_mm", "Seletar_Temperature_C", "mu_load",
                                                            "sd_load"]])

    results, r2_score = model.predict(test_data[["weekend/ph", "Punggol_Rainfall_mm",
                "Seletar_Rainfall_mm", "Seletar_Temperature_C", "mu_load",
                "sd_load"]], test_data["Fuel Efficiency KML"])
    
    y_mean = np.mean(good_train["Fuel Efficiency KML"])
    y_pred_base = np.full(len(test_data["Fuel Efficiency KML"]), y_mean)

    rmse_base = root_mean_squared_error(test_data["Fuel Efficiency KML"], y_pred_base)
    print("Baseline RMSE (training mean):", rmse_base)
    print("R2 score:", r2_score)
    # replace and test the rmse based on good data
    # compare prediction here against naive approach

    residuals = results - test_data["Fuel Efficiency KML"]
    
    kalman = Kalman()

    kalman.fit(list(residuals[:len(residuals)//2]))
    
    # --- DATABASE CONNECTION & PREPARATION ---
    # Connect to the SQLite database
    conn = sqlite3.connect('sbs.db')
    cursor = conn.cursor()

    # 1. Get Service ID (Service 5)
    cursor.execute("SELECT id FROM services WHERE service_no = 5")
    service_row = cursor.fetchone()
    if service_row:
        service_id = service_row[0]
    else:
        cursor.execute("INSERT INTO services (service_no) VALUES (5)")
        service_id = cursor.lastrowid

    # 2. Cache Bus IDs to avoid querying inside the loop
    cursor.execute("SELECT license_plate, id FROM buses")
    bus_map = {row[0]: row[1] for row in cursor.fetchall()}

    # 3. Align Test Data with the Second Half of Residuals
    cutoff = len(residuals) // 2
    subset_test_data = test_data.iloc[cutoff:]
    subset_residuals = residuals.iloc[cutoff:].values
    subset_expected = results[cutoff:]
    subset_actual = test_data["Fuel Efficiency KML"].iloc[cutoff:].values

    print("Inserting runs into database...")

    # Iterate through the second half
    for i in range(len(subset_residuals)):
        z = subset_residuals[i]
        row = subset_test_data.iloc[i]
        expected = subset_expected[i]
        actual = subset_actual[i]

        # Kalman Step
        kalman_residual, _ = kalman.step(z)
        print(kalman_residual)

        # Retrieve Context from the row
        bus_plate = row["Bus"]
        
        # Resolve Bus ID
        if bus_plate in bus_map:
            bus_id = bus_map[bus_plate]
        else:
            # If a new bus appears that isn't in DB, insert it
            cursor.execute("INSERT INTO buses (license_plate) VALUES (?)", (bus_plate,))
            bus_id = cursor.lastrowid
            bus_map[bus_plate] = bus_id

        # Construct a proper datetime string for the 'datetime' column
        # Combining the 'Date' timestamp with the 'start_time' hour
        trip_datetime = row['Date'] + pd.Timedelta(hours=int(row['start_time']))
        timestamp_str = trip_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Insert into 'trips' table
        cursor.execute('''
            INSERT INTO trips (bus_id, service_id, residual, datetime, expected, actual)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (bus_id, service_id, float(kalman_residual), timestamp_str, float(expected), float(actual)))

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database update complete.")


if __name__ == "__main__":
    main()
