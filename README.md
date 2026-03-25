# SBS Transit Hackathon 2026 Project: Fuel Performance Degradation Early Warning & Monitoring System

Done by: Goh Jun-en, Samuel and Alvin Ong Zhao Wei

## Overview

This project is designed to monitor and analyze the fuel efficiency of bus fleets. By using advanced predictive modeling and data filtering techniques, fleet managers can gain insights into gradual or sudden changes in fuel performance and take timely action for maintenance and cost optimization.

## Key Features

- **LightGBM Regression Model**: 
  - Predicts the expected fuel efficiency of bus fleets under various conditions.
  - Helps in understanding the ideal performance standard for each vehicle.

- **Anomaly Detection**:
  - Calculates the difference between predicted and actual fuel usage to highlight deviations.
  - Identifies inefficiencies and anomalies in fuel consumption.

- **Noise Smoothing with 1-D Kalman Filter**:
  - Filters out noise and fluctuations in the data for more accurate monitoring.
  - Ensures reliable detection of gradual performance degradation and sudden changes.

- **Maintenance Notifications**:
  - Plots out Trends of Fuel Performance for each bus.
  - Automated Status Updates based on gradual or sudden changes in fuel performance.
  - Supports timely maintenance to prevent costly breakdowns and improve operational efficiency.

## How It Works

1. **Data Collection and Preprocessing**:
    - Gather and preprocess operational data from buses through LTA Datamall API and NEA Weather.gov.
    - Standardize data inputs for consistency and reliability.

2. **Predictive Modeling**:
    - Train the LightGBM regression model on historical data to estimate expected fuel efficiency.

3. **Anomaly Calculation**:
    - Compare actual usage metrics with predicted outcomes.
    - Flag deviations that exceed a defined threshold.

4. **Noise Reduction**:
    - Apply the 1-D Kalman Filter to smooth the data and eliminate short-term variability.

5. **Actionable Insights**:
    - Use the filtered, cleaned data to detect trends and performance issues.
    - Generate notifications for further investigation or immediate action.