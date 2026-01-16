# Relevant libraries

import numpy as np
import pandas as pd

# Clean Relative Humidity:
    # Only keep Oct, Nov, Dec 2025 data

humid = pd.read_csv("../data/RelativeHumidityMonthlyMean.csv")
