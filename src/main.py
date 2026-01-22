from clean import clean_bus_data, fuel_efficiency_data, merge_data
from ML_model import ML_model
from kalman import Kalman
import numpy as np
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
    print(results)
    print(test_data[["Fuel Efficiency KML", "Workshop activities"]])

    residuals = results - test_data["Fuel Efficiency KML"]
    
    kalman = Kalman()

    kalman.fit(list(residuals[:len(residuals)//2]))
    # compare prediction here against naive approach
    for z in list(residuals[len(residuals)//2:]):
        print(kalman.step(z))
        # put this into sql database


if __name__ == "__main__":
    main()
