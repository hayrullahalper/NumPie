# Decoding Economic Growth: An Explainable AI Model (1960-Present)

> **A Data Science Capstone Project building a robust, interpretable machine learning pipeline to predict annual GDP growth and identify its key drivers using historical data.**

---

## 📌 Project Overview

**Problem:** Understanding economic growth is a complex, data-intensive challenge. Economic systems are non-linear and shaped by interacting domestic conditions and external forces.
**Goal:** To build a machine learning model (XGBoost) that predicts a country's **GDP Growth (annual %)** using a massive dataset spanning **1961–2023**.
**Contribution:** We merged 16 distinct datasets to create a unified panel (10,000+ rows). We addressed the challenge of missing historical data and used **Explainable AI (SHAP)** to reveal the "drivers" of growth.

### ❓ Research Questions
1. What are the most important factors ("drivers") that influence GDP growth?
2. How significant is the impact of each factor (e.g., Political Stability vs. Education)?
3. Have the most important factors changed over time (e.g., Technology in the 1990s vs. today)?

---

## 📊 Data Sources

We fused **16 datasets** into a single file. The primary data comes from the **World Bank**, augmented with global financial indicators.

### 1. Target Variable (Y)
| Code | Indicator | Link |
| :--- | :--- | :--- |
| **NY.GDP.MKTP.KD.ZG** | GDP growth (annual %) | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG) |

### 2. Feature Datasets (X)
| Code | Indicator | Link |
| :--- | :--- | :--- |
| **NE.GDI.TOTL.ZS** | Investment (% of GDP) | [World Bank](https://data.worldbank.org/indicator/NE.GDI.TOTL.ZS) |
| **NE.TRD.GNFS.ZS** | Trade (% of GDP) | [World Bank](https://data.worldbank.org/indicator/NE.TRD.GNFS.ZS) |
| **FP.CPI.TOTL.ZG** | Inflation (Consumer Prices) | [World Bank](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG) |
| **SL.UEM.TOTL.ZS** | Unemployment (% of total labor) | [World Bank](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS) |
| **NV.AGR.TOTL.ZS** | Agriculture (% of GDP) | [World Bank](https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS) |
| **EG.USE.PCAP.KG.OE** | Energy Use (kg of oil equivalent) | [World Bank](https://data.worldbank.org/indicator/EG.USE.PCAP.KG.OE) |
| **SE.XPD.TOTL.GD.ZS** | Education Spending (% of GDP) | [World Bank](https://data.worldbank.org/indicator/SE.XPD.TOTL.GD.ZS) |
| **SH.DYN.MORT** | Infant Mortality Rate | [World Bank](https://data.worldbank.org/indicator/SH.DYN.MORT) |
| **SP.POP.TOTL** | Total Population | [World Bank](https://data.worldbank.org/indicator/SP.POP.TOTL) |
| **SP.URB.TOTL.IN.ZS** | Urban Population (%) | [World Bank](https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS) |
| **IT.NET.USER.ZS** | Internet Users (%) | [World Bank](https://data.worldbank.org/indicator/IT.NET.USER.ZS) |
| **PV.EST** | Political Stability | [World Bank](https://data.worldbank.org/indicator/PV.EST) |
| **SI.POV.GINI** | Gini Index | [World Bank](https://data.worldbank.org/indicator/SI.POV.GINI) |
| **Global Market** | **Gold Price (Annual Avg)** | *External Source* |
| **Global Market** | **USD Index (DXY)** | *External Source* |

---

## 🛠️ Methodology (The Pipeline)

This project follows a rigorous data science workflow:

1.  **Data Collection:** Automated ingestion of 16 CSV files.
2.  **Data Cleaning & Merging:**
    * Converted "wide" format World Bank data to "long" format using `pandas.melt`.
    * Standardized Country Names using **ISO-3166** codes.
    * Merged all datasets into a single country-year panel.
3.  **Handling Missing Data:** Addressed significant `NaN` values (1960-1990) using logic-based imputation (e.g., Internet = 0 before 1990) and linear interpolation.
4.  **Feature Engineering:** Created temporal features:
    * `Lag_1`, `Lag_2` (Previous years' growth).
    * `Rolling_Mean_3Y` (Trend).
    * Interaction terms (e.g., Trade × Global Growth).
5.  **Modeling:**
    * **Split:** Time-aware split (Train: 1961–2014, Test: 2015–2023).
    * **Algorithms:** Benchmarked **Random Forest** vs. **XGBoost**.
6.  **Evaluation:** Used RMSE and $R^2$ Score.
7.  **Explainability:** Utilized **SHAP** to rank feature importance.
