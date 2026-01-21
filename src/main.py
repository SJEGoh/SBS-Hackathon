from clean import clean_bus_data, fuel_efficiency_data, merge_data
from ML_model import ML_model
from kalman import Kalman

def main():
    # define good values as not within 2 days of workshop day
    # split data into train + actual
    # actual store into a sql database

    data = merge_data()
    train_split = int(len(data) * 0.9)
    train_data = data[:train_split]
    test_data = data[train_split:]

    good_train = train_data[train_data["Workshop activities"] != "W"]
    print(good_train.columns)
    # this model predict fuel efficiency
    model = ML_model()

    model.fit(good_train["Fuel Efficiency KML"], good_train[["weekend/ph", "Punggol_Rainfall_mm",
                                                            "Seletar_Rainfall_mm", "Seletar_Temperature_C", "mu_load",
                                                            "sd_load"]])

    # 

if __name__ == "__main__":
    main()
