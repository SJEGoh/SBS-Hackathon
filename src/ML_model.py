import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error

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

        half_life = 5  
        w = 0.5 ** ((n - 1 - t) / half_life)  
        r = np.asarray(fuel_efficiency).reshape(-1)
        self.model.fit(features, r, sample_weight = w)
        # Look into sample weight
        self._fitted = True
        print("Model fitted!")
    
    def predict(self, test_features, test_result):
        y = self.model.predict(test_features)
        return y, root_mean_squared_error(y, test_result)
    


    




