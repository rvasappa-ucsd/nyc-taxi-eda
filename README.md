# 🚖 NYC Taxi Data: Exploratory and Predictive Modeling

Using **Apache Spark** and over a decade of NYC Yellow Taxi trip data from 2014 to 2024, this project features several phases, including large-scale exploratory data analysis, the development of a fare prediction model, and sentiment analysis of rider experiences. This comprehensive framework examines patterns in urban mobility, fare structures, and customer satisfaction by integrating structured trip records with unstructured textual feedback. The result is a comprehensive perspective on the operational and experiential dimensions of New York City's taxi ecosystem.

---

## 📌 Project Overview

This repository provides a Spark-based exploratory data analysis (EDA) of the NYC Yellow Taxi dataset. The analysis focuses on understanding:

- Trip patterns
- Fare structures
- Tipping behavior
- Operational trends
- Urban mobility insights over the past decade

---

## ⚙️ Approach

### 1. Environment Setup

- Apache Spark and PySpark configured within a Jupyter Notebook environment
- Enabled with distributed setup and runs only in SDSC Cluster

### 2. Data Engineering

- **Data Ingestion**: NYC Yellow Taxi trip data (2014–2024, all months) in Parquet format
- **Feature Engineering**:
  - Temporal features (hour, day, month, weekday/weekend)
  - Trip metrics like speed, duration, tip percentage
- **Data Cleaning**:
  - Removal of invalid trips (e.g., 0 distance/fare)
  - Filtering outliers and noisy records

### 3. Analysis Components

#### 📅 Temporal Analysis
- Hourly, daily, and monthly trip trends
- Year-over-year change tracking
- Weekday vs. weekend usage patterns

#### 💰 Financial Analysis
- Fare vs. trip distance/time correlation
- Tipping trends and tip percentage distribution
- Fare dynamics based on passenger count and trip type

#### 🚕 Trip Characteristics
- Trip speed analysis by hour
- Distance/duration categorization (short, medium, long)
- Temporal effects on trip length

#### 🗺️ Geospatial Analysis *(Planned/Future Work)*
- Pickup/dropoff clustering
- High-density zones and airport corridors

---

## 📊 Visualization Highlights

- Hourly and weekly trip distribution
- Fare vs. distance scatter plots
- Tip percentage histograms and time-of-day analysis
- Correlation heatmaps of trip features

---

## ❓ Key Exploratory Questions

### Temporal Patterns
- How do taxi usage patterns vary across time of day, day of week, and year?
- What are the busiest hours for NYC taxis?
- How has usage changed post-COVID compared to pre-pandemic years?

### Financial Insights
- What are the strongest predictors of fare amount?
- When are riders more likely to tip?
- Do trip distance and passenger count influence tips?

### Operational Insights
- What’s the average taxi speed across different times?
- How do trip durations and distances vary seasonally?
- What share of rides are short (<2 mi), medium (2–10 mi), and long (>10 mi)?

---



## 🚕📈 Model 1: Fare Prediction Model

### a. Overview 
The goal of this model is to accurately predict the fare amount for NYC Yellow Taxi rides from 2020 to 2024 based on key trip features available at the time of pickup. These include trip distance, duration, tolls, time of day, and passenger count, all of which influence how fares are calculated under NYC’s standard metered pricing rules. By leveraging Apache Spark and Dask, we trained a high-performing LightGBM regression model that handles over 150 million records. This model not only captures the complex relationships between trip variables and fare pricing but also generalizes well across a wide range of conditions and ride types.

### b. Preprocessing & Feature Engineering
To prepare the data for modeling, we implemented the following transformations:

**Filtering**

- Removed rows with fare_amount < $3 (below NYC minimum base fare) or fare_amount > $200 to eliminate invalid and extreme outliers.

![download (3)](https://github.com/user-attachments/assets/8f7abab6-b052-4ea4-8d68-a5ffa7831dcf)

_Figure 1: Histogram of fare amounts, showing common spikes at flat fares (e.g., $70 to JFK). This supports trimming extreme values._

- Removed trips with:
  - trip_distance ≤ 0 or > 35 miles (NYC city limits)
  - trip_time_minutes ≤ 0 or > 1500 (≈25 hours, extreme outliers)
    
![download (5)](https://github.com/user-attachments/assets/34cf691a-2a8c-4a80-bbf5-763464bf7d5b)
_Figure 2: Trip distance distributions from a 10% sample, segmented by mileage bins. Most trips are short, with a sharp drop-off after 3 miles. This supports the filtering of trips > 35 miles as outliers._

- Kept only records with RatecodeID = 1, representing standard metered fares (~90% of all trips).
- Clipped passenger_count to 1–5 based on NYC taxi regulations.
  
**Feature Engineering**

- Extracted temporal features from pickup time:
  - hour, dayofweek, month

- Computed trip-level efficiency metrics:
  - trip_time_minutes
  - fare_per_mile, fare_per_minute

- Excluded post-hoc features like tip_amount to prevent leakage during prediction.

After all filtering and transformations, the dataset retained 88.55% of original data (~155M rows from 175M).

### c. Modeling
We began by benchmarking a simple linear model, followed by training a more powerful tree-based model using LightGBM on a large-scale distributed infrastructure.

**🧠Linear Regression (Baseline Model)**

Our baseline model was a multivariate linear regression trained using PySpark MLlib. It used a small set of core features known to influence fare calculation:

- trip_distance: total miles traveled
- trip_time_minutes: duration of the ride
- tolls_amount: total tolls incurred
- hour: time of day the trip started

While this model provided a quick sanity check for feature importance and data health, its performance was limited due to its inability to capture non-linear relationships between variables (e.g., tipping points at certain times or distances). Residuals from this model showed underfitting, particularly in edge cases involving long trips or high tolls.

**🧠LightGBM Regressor (Final Model)**

To improve prediction accuracy, we used the LightGBM Regressor trained in a Dask environment. This allowed us to parallelize training across partitions and scale to the full dataset (over 150 million rows).

Key model characteristics:

- Framework: Dask-ML + LightGBM
- Features Used:
  - trip_distance
  - trip_time_minutes
  - tolls_amount
  - hour
- Training Configuration:
  - n_estimators = 200
  - max_depth = 15
  - learning_rate = 0.3
- Split Strategy: 80/20 stratified split based on a seeded random distribution across Dask partitions
  
We observed that trip distance and duration had the highest impact on prediction quality, consistent with NYC’s metered pricing system. Tolls also contributed significantly to variance, particularly for airport trips or bridge-heavy routes.
### d. Model Evaluation Results
The performance of both models was evaluated using standard regression metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² (Coefficient of Determination).

| Metric       | Linear Regression | LightGBM (Train) | LightGBM (Test) |
| ------------ | ----------------- | ---------------- | --------------- |
| **RMSE**     | \~5.24            | 2.66             | **2.66**        |
| **MAE**      | \~3.95            | 1.91             | **1.91**        |
| **R² Score** | \~0.74            | 0.9375           | **0.9374**      |


**Interpretation of Results:**
- LightGBM significantly outperformed the baseline linear model across all metrics, especially in reducing RMSE and MAE.
- Test RMSE of 2.66 implies that on average, the predicted fare deviates by about $2.66 from the actual fare.
  
Considering the average fare across our dataset is ~$15.34:
- RMSE is only ~17.3% of the average fare
- MAE is just ~12.5%, indicating highly precise predictions

The high R² (93.7%) means the model explains most of the variance in fare pricing, making it reliable even in real-world deployment scenarios.

### e. Model Fit & Residual Analysis
To assess generalization and identify possible weaknesses, we analyzed the model’s residuals:
- Low variance: Training and test metrics were nearly identical, indicating no overfitting
- Moderate bias: The model occasionally underpredicts fares for:
  - Very long trips
  - Trips at atypical hours
  - Flat-fare scenarios (e.g., airports)
- Overall error bounds: With a test MAE of ~$1.91, most predictions were within reasonable, real-world acceptable ranges

![download (4)](https://github.com/user-attachments/assets/c946c92b-066c-4e7d-b940-17ff9f76cc07)
_Figure 3: Residuals plotted against trip distance, time, and hour reveal slight underprediction on long or irregular trips, but overall consistency._

### f. Opportunities for Improvement
- Time-Aware Modeling:
NYC fare structures change year-to-year. Training on only the most recent 1–2 years may capture current fare rules more accurately.

- Incorporate External Features:
Weather, traffic, and special events can influence both fare and trip time. These could help reduce errors on long or delayed trips.

- Geospatial Modeling:
Including pickup/dropoff zone clusters may help better account for flat-rate zones and congestion effects.

### g. Final Thoughts
Our tuned LightGBM regressor (Test RMSE: $2.66, MAE: $1.91, R²: 0.9371) delivers reliable, production-grade fare estimates. It captures both linear distance-fare trends and nonlinear effects like tolls and surcharges with minimal overfitting. With an RMSE of approximately ±17.3% and MAE of ±12.5% relative to the $15.34 average fare, the model provides a robust, generalizable solution for NYC taxi-fare prediction.

As an additional validation step, we trained a version of the model using only 2019–2023 data and tested it on 2024. Despite never seeing 2024 data during training, it achieved similar performance (RMSE: 2.98, MAE: 1.88, R²: 0.9215). This further supports the model’s ability to generalize to future conditions using its current features and tuning.

---
## 📂 Repository Structure

```text
nyc-taxi-eda/
├── nyc_taxi_eda.ipynb        # Main Jupyter Notebook for Spark-based EDA
├── nyc_taxi_data/            # Folder for downloaded Parquet trip data
├── model_1.ipynb             # Main notebook for Model 1: Fare Prediction (LightGBM)
├── model_1.pkl               # Saved LightGBM model from model_1.ipynb
├── model_1_test24.ipynb      # Notebook retraining model on 2019–2023, tested on 2024
├── model_1_test24.pkl        # Saved model trained on 2019–2023 data
├── README.md                 # Project overview and documentation
└── requirements.txt          # Python dependencies (optional)
```

### EDA in SDSC Cluster

EDA: https://github.com/rvasappa-ucsd/nyc-taxi-eda/blob/main/nyc_taxi_eda.ipynb

### Model 1, Fare Prediction Model

Model 1: https://github.com/rvasappa-ucsd/nyc-taxi-eda/blob/Milestone3/model_1.ipynb

### Model 2, Sentiment Analysis

Model 2: coming soon

---

## 📎 Data Source

This project uses publicly available NYC Yellow Taxi data published by the NYC Taxi & Limousine Commission (TLC).

- **Official TLC page**:  
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

- **Parquet file archive (2014–2024)**:  
  All monthly files accessed via CloudFront CDN:  
  ```
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_[2014-2024].[01-12].parquet
  ```

- **File Format**: Parquet (columnar, efficient for Spark)

---

## 📈 Future Enhancements

- Interactive dashboard using Plotly Dash
- Advanced geospatial analysis using H3 or GeoPandas
- Prediction Models using ML for predicting Tips/Fares/Future usage patterns, expected demand etc

---

## 👩‍💻 Authors

This project is developed by students at **UC San Diego** as part of 232-R Group Project Spring Semester:

- **Harsh Arya** — harya@ucsd.edu  
- **Gabrielle Despaigne** — gdespaigne@ucsd.edu  
- **Zack Mosley** — zmosley@ucsd.edu  
- **Camila Paik** — capaik@ucsd.edu  
- **Raghav Vasappanavara** — rvasappanavara@ucsd.edu

---

## 📄 License

This repository is for educational and research purposes. Data is publicly available and governed by NYC TLC data usage policies.

---
