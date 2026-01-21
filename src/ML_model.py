import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

class ML_model:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=800,
            min_samples_leaf=20,
            l2_regularization=0.2,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=42,
        )
        self._fitted = False

    def fit(self, fuel_efficiency, features):
        r = np.asarray(fuel_efficiency).reshape(-1)
        self.model.fit(features, r)
        # Look into sample weight
        self._fitted = True
        print("Model fitted!")
    
    def predict(self, test_features):
        return self.model.predict(test_features)

    




