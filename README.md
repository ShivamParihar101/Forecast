# ⚡ Electricity Demand Forecasting

## Project Overview

This project focuses on forecasting hourly electricity demand using time-series forecasting and machine learning techniques. The goal is to analyze historical electricity consumption patterns and predict future demand for a user-selected date, time, and forecast duration.

The project includes data exploration, preprocessing, feature engineering, model development, hyperparameter tuning, model evaluation, and a Streamlit web application for interactive forecasting.

## Dataset

The project uses the **PJMW_MW Hourly Electricity Demand Dataset**.

- **Target variable:** PJMW_MW
- **Frequency:** Hourly
- **Original dataset size:** 143,206 observations
- **Dataset contains:** Date/time information and electricity demand measurements

### Data Preprocessing

The following preprocessing steps were performed:

- Converted timestamps into datetime format.
- Sorted the dataset chronologically.
- Handled duplicate timestamps by averaging electricity demand values.
- Created a complete hourly time index.
- Handled missing hourly values using time interpolation.
- Created calendar-based features.
- Created cyclical time features.
- Created lag and rolling statistical features.
- Split the dataset chronologically into training, validation, and testing sets.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand electricity demand patterns.

The analysis included:

- Electricity demand over time
- Hourly average demand
- Monthly average demand
- Weekday vs weekend demand
- Distribution of electricity demand
- Boxplots for demand analysis

The analysis showed patterns related to time of day, weekdays, weekends, and seasonal variations.

## Models Developed

The following models were implemented and evaluated:

1. Naive Forecasting
2. Seasonal Naive Forecasting
3. ARIMA
4. SARIMA
5. Random Forest
6. XGBoost
7. LSTM

Hyperparameter tuning was performed for selected models to improve forecasting performance.

## Model Evaluation

Models were evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
- **MAPE (Mean Absolute Percentage Error):** Measures prediction error as a percentage.

Models were evaluated across multiple forecast horizons, including:

- 1 hour
- 3 hours
- 6 hours
- 8 hours
- 24 hours
- 43 hours
- 48 hours
- 168 hours

## Final Model

Based on the final test evaluation, the **Tuned SARIMA model** was selected for deployment in the Streamlit application.

### Final Test Performance

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---:|---:|---:|
| Tuned SARIMA | 145.04 | 178.18 | 2.84% |
| Tuned XGBoost | 170.19 | 217.05 | 3.19% |
| Tuned Random Forest | 312.36 | 338.31 | 6.29% |

The Tuned SARIMA model achieved the best average performance across the evaluated forecast horizons.

## Streamlit Application

The project includes an interactive web application built using Streamlit.

Users can:

- Select a forecast start date.
- Select a forecast start time.
- Specify the number of future hours to forecast.
- Generate electricity demand predictions.
- View forecast summary statistics.
- Visualize the forecast graph.
- View historical electricity demand separately.
- View forecast results in a table.
- Download forecast results as a CSV file.

The application supports user-defined forecast horizons.

## Project Structure

```text
Forecast/
│
├── app.py
├── forecast.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── PJMW_MW_Hourly.xlsx
│
├── artifacts/
│   ├── processed_data.csv
│   ├── final_training_history.csv
│   ├── final_test_results.csv
│   ├── final_model_comparison.csv
│   │
│   └── models/
│       ├── final_sarima_model.pkl
│       └── forecast_metadata.pkl
│
└── src/
```

## Installation

### 1. Clone or download the project

Download the project repository to your computer.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the following command from the project directory:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Limitations

- The model was trained using historical electricity demand data ending in 2016.
- Forecasts for dates far beyond the training period should be interpreted cautiously.
- Forecast uncertainty generally increases as the forecast horizon becomes longer.
- The application demonstrates forecasting functionality using the trained historical model and does not incorporate newly observed electricity demand data.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- XGBoost
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook

## Conclusion

This project demonstrates a complete time-series forecasting workflow, from exploratory data analysis and preprocessing to model development, evaluation, and deployment.

The final Tuned SARIMA model was selected based on its performance on the test dataset and integrated into an interactive Streamlit application that allows users to generate and visualize electricity demand forecasts.