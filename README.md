# SBS-Hackathon

1. Feature engineering (use existing datasets)
2. Feed into XGBoost model 
- Why XGBoost?
    No need for scaling, less overfitting than an average decision tree
    Able to handle less data than deep learning models
Output: Expected fuel efficiency given conditions
3. Calculate residual, and normalise using Kalman filter


Ok wtf now
1. Decompose to per route
2. Find expected passenger load + variance per route using data alvin cleaned (thanks alvin :)
3. Feed that in, along with expected variance rainfall + other weather conditions?
    Why is weather aggregated by month????
4. Feed in daily weather data

