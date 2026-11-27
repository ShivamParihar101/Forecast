# ============================================
# Electricity Demand Forecasting Application
# ============================================

from huggingface_hub import hf_hub_download
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
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

MODEL_PATH = BASE_DIR / "artifacts" / "models" / "final_sarima_model.pkl"

METADATA_PATH = BASE_DIR / "artifacts" / "models" / "forecast_metadata.pkl"

HISTORY_PATH = BASE_DIR / "artifacts" / "final_training_history.csv"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="shivamparihar069/electricity-demand-sarima",
        filename="final_sarima_model.pkl"
    )

    model = SARIMAXResults.load(model_path)
    return model


@st.cache_data
def load_metadata():

    metadata = joblib.load(METADATA_PATH)

    return metadata


@st.cache_data
def load_historical_data():

    historical_data = pd.read_csv(
        HISTORY_PATH,
        index_col=0,
        parse_dates=True
    )

    return historical_data


final_model = load_model()

forecast_metadata = load_metadata()

historical_data = load_historical_data()


# ============================================
# 3. APPLICATION TITLE
# ============================================

st.title("⚡ Electricity Demand Forecasting")

st.write(
    """
    This application forecasts future hourly electricity demand
    using a trained SARIMA time-series forecasting model.

    Enter the number of future hours you want to predict.
    """
)


# ============================================
# 4. USER INPUT
# ============================================

st.subheader("Forecast Settings")

col1, col2, col3 = st.columns(3)

with col1:
    forecast_date = st.date_input(
        "Forecast Start Date",
        min_value=pd.Timestamp.now().date(),
        value=pd.Timestamp.now().date()
    )

with col2:
    forecast_time = st.time_input(
        "Forecast Start Time",
        value=pd.Timestamp.now().time().replace(minute=0, second=0, microsecond=0)
    )

with col3:
    forecast_horizon = st.number_input(
        "Forecast Duration (Hours)",
        min_value=1,
        value=24,
        step=1
    )


# ============================================
# 5. FORECAST FUNCTION
# ============================================

def forecast_electricity_demand(start_datetime, horizon):

    if horizon <= 0:
        raise ValueError("Forecast horizon must be greater than zero.")

    # Get current datetime
    current_datetime = pd.Timestamp.now().floor("h")

    # Calculate how many hours ahead the selected start is
    start_datetime = pd.Timestamp(start_datetime).floor("h")

    hours_ahead = int(
        (start_datetime - current_datetime).total_seconds() / 3600
    )

    if hours_ahead < 0:
        raise ValueError("Forecast start time cannot be in the past.")

    # Forecast enough hours to reach the selected date/time
    total_steps = hours_ahead + int(forecast_horizon)

    predictions = final_model.forecast(steps=total_steps)

    # Select predictions corresponding to the requested period
    selected_predictions = np.asarray(predictions)[
        hours_ahead:hours_ahead + int(forecast_horizon)
    ]

    future_dates = pd.date_range(
        start=start_datetime,
        periods=int(forecast_horizon),
        freq="h"
    )

    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_PJMW_MW": selected_predictions
    })

    return forecast_df


# ============================================
# 6. GENERATE FORECAST BUTTON
# ============================================

if st.button("Generate Forecast", type="primary"):

    with st.spinner("Generating electricity demand forecast..."):
        selected_datetime = pd.Timestamp.combine(forecast_date, forecast_time)
        forecast_df = forecast_electricity_demand(selected_datetime,int(forecast_horizon))
        
    st.success("Forecast generated successfully!")


    # ========================================
    # 7. DISPLAY FORECAST SUMMARY
    # ========================================

    st.subheader("Forecast Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Forecast Hours",
            len(forecast_df)
        )

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


    # ==========================================
    # FORECAST VISUALIZATION
    # ==========================================

    st.subheader("Electricity Demand Forecast")

    # Plot ONLY the forecast period
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        forecast_df["Datetime"],
        forecast_df["Predicted_PJMW_MW"],
        color="red",
        linestyle="--",
        marker="o",
        markersize=3,
        label="Forecasted Demand"
    )

    ax.set_title("Future Electricity Demand Forecast")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Electricity Demand (MW)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================
    # HISTORICAL DEMAND (SEPARATE GRAPH)
    # ==========================================

    with st.expander("View Historical Electricity Demand"):

        historical_to_plot = historical_data.tail(168)

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            historical_to_plot.index,
            historical_to_plot["PJMW_MW"],
            color="blue",
            label="Historical Demand"
        )

        ax.set_title("Historical Electricity Demand (Last 7 Days)")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Electricity Demand (MW)")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ========================================
    # 9. DISPLAY FORECAST TABLE
    # ========================================

    st.subheader("Forecast Data")

    display_forecast = forecast_df.copy()

    display_forecast["Datetime"] = (
        display_forecast["Datetime"]
        .dt.strftime("%Y-%m-%d %H:%M")
    )

    display_forecast["Predicted_PJMW_MW"] = (
        display_forecast["Predicted_PJMW_MW"]
        .round(2)
    )

    st.dataframe(
        display_forecast,
        use_container_width=True
    )


    # ========================================
    # 10. DOWNLOAD FORECAST
    # ========================================

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name="electricity_demand_forecast.csv",
        mime="text/csv"
    )