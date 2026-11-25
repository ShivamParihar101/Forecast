# ============================================
# Electricity Demand Forecasting Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from huggingface_hub import hf_hub_download
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# ============================================
# 1. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)


# ============================================
# 2. LOAD MODEL AND DATA
# ============================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "artifacts" / "models" / "forecast_metadata.pkl"
HISTORY_PATH = BASE_DIR / "artifacts" / "final_training_history.csv"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="shivamparihar069/electricity-demand-sarima",
        filename="final_sarima_model.pkl"
    )
    return SARIMAXResults.load(model_path)


@st.cache_data
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_data
def load_historical_data():
    return pd.read_csv(
        HISTORY_PATH,
        index_col=0,
        parse_dates=True
    )


# Show a helpful message if model or files fail to load
try:
    final_model = load_model()
    forecast_metadata = load_metadata()
    historical_data = load_historical_data()
except Exception as error:
    st.error("The model or required data files could not be loaded.")
    st.exception(error)
    st.stop()


# ============================================
# 3. APPLICATION TITLE
# ============================================

st.title("⚡ Electricity Demand Forecasting")

st.write(
    """
    This application forecasts future hourly electricity demand
    using a trained SARIMA time-series forecasting model.

    Enter how many hours ahead you want to forecast.
    Forecasts begin immediately after the last timestamp in the
    historical dataset used to train the model.
    """
)


# ============================================
# 4. USER INPUT
# ============================================

st.subheader("Forecast Settings")

forecast_horizon = st.number_input(
    "Forecast Horizon (Hours)",
    min_value=1,
    value=24,
    step=1
)

last_datetime = pd.Timestamp(forecast_metadata["last_datetime"])
forecast_start = last_datetime + pd.Timedelta(hours=1)

st.info(
    f"Forecast will start from: **{forecast_start.strftime('%Y-%m-%d %H:%M')}** "
    f"(based on the last historical timestamp: {last_datetime.strftime('%Y-%m-%d %H:%M')})"
)


# ============================================
# 5. FORECAST FUNCTION
# ============================================

def forecast_electricity_demand(horizon):
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
        raise ValueError("Forecast horizon must be a positive integer.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be greater than zero.")

    last_datetime = pd.Timestamp(forecast_metadata["last_datetime"])

    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=int(horizon),
        freq="h"
    )

    predictions = final_model.forecast(steps=int(horizon))

    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_PJMW_MW": np.asarray(predictions)
    })

    return forecast_df


# ============================================
# 6. GENERATE FORECAST
# ============================================

if st.button("Generate Forecast", type="primary"):

    try:
        with st.spinner("Generating electricity demand forecast..."):
            forecast_df = forecast_electricity_demand(
                int(forecast_horizon)
            )

        st.success("Forecast generated successfully!")

        # ========================================
        # 7. FORECAST SUMMARY
        # ========================================

        st.subheader("Forecast Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Forecast Hours", len(forecast_df))

        with col2:
            st.metric(
                "Average Predicted Demand",
                f"{forecast_df['Predicted_PJMW_MW'].mean():,.2f} MW"
            )

        with col3:
            st.metric(
                "Maximum Predicted Demand",
                f"{forecast_df['Predicted_PJMW_MW'].max():,.2f} MW"
            )

        # ========================================
        # 8. FORECAST VISUALIZATION
        # ========================================

        st.subheader("Electricity Demand Forecast")

        fig, ax = plt.subplots(figsize=(14, 6))

        historical_to_plot = historical_data.tail(168)

        ax.plot(
            historical_to_plot.index,
            historical_to_plot["PJMW_MW"],
            color="blue",
            label="Historical Demand"
        )

        ax.plot(
            forecast_df["Datetime"],
            forecast_df["Predicted_PJMW_MW"],
            color="red",
            linestyle="--",
            marker="o",
            markersize=3,
            label="Forecasted Demand"
        )

        ax.axvline(
            x=forecast_df["Datetime"].iloc[0],
            color="black",
            linestyle=":",
            label="Forecast Start"
        )

        ax.set_title("Historical and Forecasted Electricity Demand")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Electricity Demand (MW)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ========================================
        # 9. FORECAST TABLE
        # ========================================

        st.subheader("Forecast Data")

        display_forecast = forecast_df.copy()
        display_forecast["Datetime"] = display_forecast["Datetime"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        display_forecast["Predicted_PJMW_MW"] = (
            display_forecast["Predicted_PJMW_MW"].round(2)
        )

        st.dataframe(
            display_forecast,
            use_container_width=True
        )

        # ========================================
        # 10. DOWNLOAD FORECAST
        # ========================================

        csv_data = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name="electricity_demand_forecast.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error("Forecast generation failed.")
        st.exception(error)
