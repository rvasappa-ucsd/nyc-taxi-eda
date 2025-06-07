# 🚖 NYC Taxi Data: Exploratory and Predictive Modeling
New York City has one of the most complex transportation networks in the world, making urban mobility a vital concern for the millions of residents who rely on affordable and reliable transit each day. With the rise of app-based ride-share alternatives, traditional taxi services have experienced a noticeable decline in ridership - a trend that has drawn increasing media attention over the past decade. As such, understanding how fare structures, tipping patterns, and service availability influence rider behavior is essential for identifying consumer trends and shaping future transportation policy. Through this project, we aim to provide data-driven insights that can support more equitable pricing strategies, improve service delivery, and enhance commuter satisfaction across New York City's five boroughs.

Using **Apache Spark** and over a decade of NYC Yellow Taxi trip data from 2014 to 2024, this project features several phases, including large-scale exploratory data analysis (EDA) and the development of a fare prediction model. This comprehensive framework examines patterns in urban mobility, fare structures, and customer satisfaction by integrating structured trip records with unstructured textual feedback. The result is a multi-dimensional perspective on the operational and experiential dimensions of New York City's taxi ecosystem. By building a predictive model for taxi fares and uncovering usage patterns over time, our goal was to demonstrate how large-scale data can be translated into actionable insights for transportation stakeholders and urban planners.

From a broader perspective, this project highlights the value of scalable data science tools in addressing complex real-world problems. Predictive modeling enables more informed decisions about resource allocation, dynamic pricing, and long-term planning. By analyzing historical ride data at scale, we demonstrate how big data infrastructure like **Apache Spark** can uncover patterns that support the design of more efficient and equitable urban transit systems. This approach provides a foundation for evidence-based policy and operational improvements within New York City's evolving mobility landscape.

---

## ⚙️ Exploratory Data Analysis

### a. Overview
As part of drafting the initial scope of research for this project, the group conducted exploratory data analysis (EDA) on the NYC Yellow Taxi dataset, focusing on trip records from 2014 to 2024. The purpose of this analysis was to surface patterns, anomalies, and emerging themes that could guide the formulation of relevant research questions. By examining trip frequency, fare structures, tipping behavior, and operational trends across time and geography, we developed a foundational understanding of urban mobility dynamics. These insights directly informed the later stages of the project, including fare prediction modeling and sentiment analysis.

### b. Environment Setup
Because the dataset is relatively large (>12GB), initial exploration was conducted using **Apache Spark** and **PySpark**, configured within a Jupyter Notebook environment running on the SDSC Cluster with a distributed setup.

### c. Data Engineering

- **Data Ingestion**: NYC Yellow Taxi trip data (2014–2024, all months) in Parquet format
- **Feature Engineering**:
  - Temporal features (hour, day, month, weekday/weekend)
  - Trip metrics like speed, duration, tip percentage
- **Data Cleaning**:
  - Removal of invalid trips (e.g., 0 distance/fare)
  - Filtering outliers and noisy records

### Analysis Components

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

## 📊 Visualization Highlights

- Hourly and weekly trip distribution
- Fare vs. distance scatter plots
- Tip percentage histograms and time-of-day analysis
- Correlation heatmaps of trip features

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

## 🚕📈 Model 2: Sentiment Analysis Model

### a. Overview 
The goal of this model is to accurately predict sentiment by way of tip percentage for NYC Yellow Taxi rides from 2020 to 2024 based on key trip features. Though we never got to train the model on the cluster a smaller scale test version of the model was trained and tested on the 2023 dataset.

### b. Preprocessing & Feature Engineering
To ensure data quality and minimize the impact of erroneous or extreme outlier values, we applied a strict set of filters.

- **Fare Amount:** Must be greater than \$3 (NYC minimum base fare) and less than \$200.
- **Tip Amount:** Cannot be negative.
- **Tip Percentage (`tip_pct`):** Must be between 0 and 100.
- **RatecodeID:** Only standard metered trips (RatecodeID = 1) were included.
- **Trip Time (`trip_time_minutes`):** Must be between 0 and 800 minutes.
- **Tolls Amount:** Must be between \$0 and \$50.
- **Trip Distance:** Must be greater than 0 and less than 35 miles.

These filters remove invalid, erroneous, or outlier records, ensuring that our analyses and models reflect realistic NYC taxi operations. 
| **Filter**        | **Rows Before** | **Rows After** | **% Remaining** | **% Removed** |
|-------------------|-----------------|---------------|-----------------|--------------|
| Distance Filter   | 3,066,766       | 2,785,198     | 90.82%          | 9.18%        |

![Image](https://github.com/user-attachments/assets/dfc5c2d0-6a88-4f64-8d1a-c91a79456c57)
Figure 4: Histogram of tip percentages of fare. (0% tip accounted for about 21% of total data volume)


As part of the preprocessing pipeline, we engineered several new features to better capture temporal patterns, ride dynamics, and rider behavior. The following features were added to the dataset:

- **tip_pct**: Tip percentage of fare amount (`tip_amount` / `fare_amount` * 100)
- **trip_time_minutes**: Total trip duration in minutes, calculated from pickup and dropoff timestamps
- **pickup_hour**: Hour of day when the trip started (0–23)
- **hour_sin, hour_cos**: Sine and cosine transforms of the pickup hour, encoding the cyclical nature of time (useful for machine learning models)
- **day_of_week**: Day of week of pickup (1=Sunday, 7=Saturday)
- **is_weekend**: Indicator for whether the trip occurred on a weekend (1 if Saturday or Sunday, 0 otherwise)
- **fare_per_min**: Fare amount divided by trip duration, indicating cost per minute
- **fare_per_mile**: Fare amount divided by trip distance, indicating cost per mile

These engineered features help capture important patterns in the data (e.g., time-of-day effects, weekday vs. weekend dynamics, fare efficiency) and enhance the performance of downstream predictive models.

![Image](https://github.com/user-attachments/assets/903e8d9a-0394-4501-a5ed-7726ddaf0b55)
Figure 5: Feature importance

The bump at around 10 miles was investigated. Within this range the most common pickup and drop off location was LaGuardia Airport which epxlains the hike in tip percentages

### c. Modeling

Originally we tried to use a regressor to predict tip percentage directly but could not achieve an RMSE less than 8.5 which when binned in no-tip, low-tip, and high-tip produced unsatisfactory results. Instead we opped for a direct classifier.


For the second phase of our analysis, we focused on predicting **tipping behavior** by categorizing rides into three classes based on tip percentage:  
- No tip  
- Low tip  
- High tip

This approach enables us to interpret rider sentiment and generosity, using tip percentage as a proxy for satisfaction.

**Class Definition & Preprocessing**

- We sampled 10% of the filtered dataset to enable efficient model development.
- Only nonzero tip rides were considered for splitting, with the median (50th percentile) tip percentage—**26.61%**—used as a cutoff.
- To ensure clear class boundaries, we dropped a 2% “gray zone” around the median.
- Final classes:
    - **0:** No tip
    - **1:** Low tip (tip_pct between 0 and just below the median)
    - **2:** High tip (tip_pct above the median)

**Feature Engineering & Inputs**

The model leveraged both raw and engineered features to capture the ride context and passenger characteristics:
- Numerical: `passenger_count`, `trip_distance`, `trip_time_minutes`, `extra`, `tolls_amount`, `congestion_surcharge`, `airport_fee`, `fare_per_min`, `fare_per_mile`, `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`
- Categorical: `payment_type`, `VendorID` (one-hot encoded)

**Model Training**

We trained a multiclass LightGBM classifier on the engineered dataset, using a **stratified 80/20 train/validation split**. The model used balanced class weights to address class imbalance and optimize for multi-class log loss.

- **Framework:** LightGBM (scikit-learn interface)
- **Objective:** Multiclass classification (3 classes)
- **Class weighting:** Balanced



### d. Model Evaluation Results
#### Sentiment Bucketization (Tip Percentage)

- **Cutpoint for bucket (nonzero tips):**  
  Median = 26.61%

- **Rows after dropping 2% gap around median:**  
  249,383

**Bucket distribution (% of kept rides):**

| Sentiment Class | % of Rides   |
|:---------------:|:------------:|
| 0               | 23.90%       |
| 1               | 37.43%       |
| 2               | 38.68%       |



#### 3-Class Model Accuracy

- **Overall accuracy:** 0.794

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.87   | 0.93     | 11,923  |
| 1     | 0.77      | 0.70   | 0.73     | 18,761  |
| 2     | 0.72      | 0.84   | 0.77     | 19,193  |

| Metric         | Value |
|----------------|-------|
| Accuracy       | 0.79  |
| Macro avg F1   | 0.81  |
| Weighted avg F1| 0.80  |



### e. Opportunities for Improvement

Due to resource constraints, we were unable to train the tip percentage classification model on the full dataset using the SDSC cluster. Our current results are based on a 10% sample and should be viewed as a proof of concept. Future work could focus on scaling the modeling pipeline for distributed training, incorporating additional external features (such as weather or event data), and refining class boundaries for even more robust sentiment prediction.



### f. Final Thoughts

Our LightGBM-based tip percentage classifier demonstrates that rider tipping sentiment can be predicted with high accuracy using features derived from trip details and passenger context. Achieving an overall validation accuracy of 79% and strong precision/recall across all sentiment classes, this model serves as a proof of concept for real-time rider sentiment analysis. Despite being trained on a 10% data sample, the results indicate clear separability between non-tippers, low tippers, and high tippers.

Scaling this approach to the full dataset and cluster environment, as well as incorporating additional features, could further improve robustness and generalizability. Nonetheless, these results highlight the value of using interpretable, engineered features for classifying rider sentiment foundational for future work in feedback prediction, targeted service improvements, and data-driven policy insights.





---

## 💬 Discussion

The final LightGBM model captured key fare drivers well: trip distance and time explained most of the variance, and tolls boosted accuracy for airport and bridge-heavy trips. Residual analysis showed:

- Low variance between train/test → no overfitting
- Slight bias on long trips or flat-fare zones
- Still within acceptable real-world prediction bounds

### Shortcomings

- No geospatial features (e.g., pickup/dropoff zones)
- Did not integrate weather/event data
- Flat-fare and irregular long trips slightly underpredicted

## ✅ Conclusion

Analyzing over a decade of NYC Yellow Taxi data offered valuable insights into the complexity and variability of real-world transportation systems. While the dataset was consistent in structure, it required substantial preprocessing to address outliers, data quality issues, and nuances such as flat-fare policies and non-standard trips. These challenges emphasized the importance of thoughtful data engineering when working with large-scale, semi-structured data.

Patterns in rider behavior, such as tipping trends and fare distributions, were both predictable and contextually influenced, highlighting how human factors and city dynamics shape mobility data. The process of cleaning, transforming, and modeling the data required not only technical skill but also real-world knowledge and awareness. 

New York City’s geography, commuter behavior, traffic patterns, and fare policies all directly affect the structure and variability of this data. Ultimately, this dataset provided an opportunity to bridge data science with urban analytics, and it highlighted the importance of scalable tools, careful validation, and critical thinking in developing predictive solutions for real-world applications in transportation and urban planning. 

If this project were extended, future work would focus on incorporating sentiment analysis, external factors such as weather and traffic congestion, as well as geospatial clustering for capturing location specific fare behavior. With additional content-awareness features, the model we developed could evolve into a more sophisticated tool capable of informing not just pricing but also policy, operations, and commuter equity in real time transit systems.

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

## 👩‍💻 Authors

This project is developed by students at **UC San Diego** as part of 232-R Group Project Spring 2025 Semester. Each student within the group shared responsibilities on every milestone of the project, including feedback, model building, review, and analysis. The main collaborations made towards this project is as follows:

- **Harsh Arya** (harya@ucsd.edu) - Created Model 2, sentiment analysis of fare tipping. Reviewed initial abstracts, models, and writeups before each submission. 
- **Gabrielle Despaigne** (gdespaigne@ucsd.edu) - Reviewed initial abstracts, exploratory data analysis, drafted introduction section on readme, final edits and review of readme document before submission.
- **Zack Mosley** (zmosley@ucsd.edu) - Created Model 1, fare prediction model. Provided readme write-up of model performance and results, reviewed models and writeups before each submission. 
- **Camila Paik** (capaik@ucsd.edu) - Wrote abstract, tested initial exploratory XGB model for fare prediction, drafted model 1 and project conclusion writeup on readme document, and submitted milestones on behalf of the group.
- **Raghav Vasappanavara** (rvasappanavara@ucsd.edu) - Created GitHub repo, headed Exploratory Data Analysis and provided assistance with development of Models 1 and 2. Reviewed models and writeups before submission. 

---

## 📄 License

This repository is for educational and research purposes. Data is publicly available and governed by NYC TLC data usage policies.

---
