import numpy as np
import pandas as pd

class Kalman:
    def __init__(self):
        self.Q = 0
        self.R = 0
        self.x = 0
        self.P = 0
        self.fitted = False

    def fit(self, residuals):
        self.R = np.var(residuals - pd.Series(residuals).rolling(5, center=True).mean())
        self.x = residuals[0]
        self.P = self.R * 100
        self.fitted = True

    def predict(self):
        self.P = self.P + self.Q
    
    def update(self, z):
        K = self.P/(self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P

        return self.x, self.P
    
    def step(self, z):
        self.predict()
        return self.update(z)
        
