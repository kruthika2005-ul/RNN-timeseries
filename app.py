import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

with st.sidebar:

    st.title("Airline Forecasting")

    future_months = st.slider(
        "Forecast Months",
        min_value=1,
        max_value=24,
        value=12
    )

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

loader = DataLoader(r"data\airline-passengers.csv")
df = loader.load_data()

# ------------------------------------------------
# Header
# ------------------------------------------------

st.title("✈️ Airline Passenger Forecasting using RNN")

st.write(
    "Forecast future airline passenger counts using a Recurrent Neural Network."
)

# ------------------------------------------------
# Model Metrics
# ------------------------------------------------

st.header("Model Evaluation")

mae, mse, rmse = Evaluator().evaluate()

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

# ------------------------------------------------
# Dataset
# ------------------------------------------------

st.header("Dataset")

st.dataframe(df)

# ------------------------------------------------
# Historical Trend
# ------------------------------------------------

st.header("Historical Passenger Trend")

fig = px.line(
    df,
    x=df.index,
    y="total_passengers",
    title="Historical Passenger Count"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecast
# ------------------------------------------------

st.header("Future Forecast")

if st.button("Generate Forecast"):

    with st.spinner("Generating Forecast..."):

        forecaster = Forecaster()

        future = forecaster.forecast(future_months)

    last_date = df.index[-1]

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_months,
        freq="MS"
    )

    forecast_df = pd.DataFrame({

        "Month": future_dates,

        "Predicted Passengers": future.flatten()

    })

    st.success("Forecast Generated Successfully.")

    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv"
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df["total_passengers"],
            mode="lines",
            name="Historical"
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=forecast_df["Month"],
            y=forecast_df["Predicted Passengers"],
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig2.update_layout(
        title="Historical vs Forecast",
        xaxis_title="Month",
        yaxis_title="Passengers"
    )

    st.plotly_chart(fig2, use_container_width=True)