# Edge AI Anomaly Detection in IoT Environments

## Overview
This repository contains the code and documentation for my thesis project on **IoT Anomaly Detection**. The system monitors indoor/outdoor temperature dynamics and utilizes Machine Learning at the Edge to detect unnatural thermal shocks (e.g., sudden window openings in winter) while ignoring natural temperature gaps caused by room insulation.

## Architecture
1. **Hardware:** Arduino MKR WiFi 1010 + Oplà IoT Kit.
2. **Data Pipeline:** - Indoor temperature from HTS221 sensor.
   - Outdoor temperature via OpenWeatherMap API.
   - 10-minute resampling interval for maximal energy efficiency (Deep Sleep).
3. **Machine Learning (Isolation Forest):**
   - **Feature Engineering:** Absolute temperatures are replaced by dynamic features (`roc_10min`, `delta_thermal_gap`, `acceleration`) to avoid false positives caused by building insulation.
   - **Scaling:** `RobustScaler` is used to prevent outliers from skewing the normalization process.
4. **Edge Deployment:** The trained Isolation Forest model is approximated via a lightweight linear proxy scorer running entirely on-device (C++), requiring minimal RAM and avoiding continuous cloud dependency. Anomalies trigger an active Wi-Fi connection to push alerts to **ThingSpeak**.

## Repository Structure
- `/data`: Contains the `feeds.csv` raw dataset (1-minute intervals).
- `/src/python_analysis`: Python scripts for data resampling, model training, feature extraction, and the generation of comparison metrics (Exp1 to Exp4).
- `/src/arduino_edge`: C++ firmware for the Arduino MKR 1010, including the on-device ring buffer, scaling logic, proxy scorer, and ThingSpeak integration.
- `/presentations`: Slide decks explaining the core problem, methodology, and experimental results.

## Key Findings
Through comparative experiments, the implementation of Feature Engineering and Robust Scaling more than doubled the model's anomaly **Score Separation** (from 0.129 in raw data to 0.280), effectively isolating true anomalies from normal environmental fluctuations.
