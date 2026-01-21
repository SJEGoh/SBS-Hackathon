import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

class ML_model:
    def __init__(self):

        self.model = ExtraTreesRegressor(
            n_estimators=500,        
            max_depth=None,         
            min_samples_leaf=5,      
            min_samples_split=10,
            max_features="sqrt",     
            bootstrap=False,         
            n_jobs=-1,
            random_state=42,
        )
        self._fitted = False

    def fit(self, fuel_efficiency, features):
        n = len(fuel_efficiency)
        t = np.arange(n)

        half_life = 5  # e.g. 30 trips/days until weight halves
        w = 0.5 ** ((n - 1 - t) / half_life)  # newest weight = 1.0
        r = np.asarray(fuel_efficiency).reshape(-1)
        self.model.fit(features, r, sample_weight = w)
        # Look into sample weight
        self._fitted = True
        print("Model fitted!")
    
    def predict(self, test_features):
        return self.model.predict(test_features)

    




