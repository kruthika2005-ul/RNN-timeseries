import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.evaluate import Evaluator
from src.forecast import Forecaster

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:

    st.image("assets/airplane.png", width=150)

    st.title("✈️ Airline Forecast")

    future_months = st.slider(
        "Forecast Months",
        min_value=1,
        max_value=24,
        value=12
    )

    st.markdown("---")

    st.info(
        "Forecast future airline passenger demand using an LSTM model."
    )

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    font-size:42px;
    font-weight:800;
    color:#1565C0;
}

.subtitle{
    font-size:18px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()

# ---------------------------------
# Header
# ---------------------------------
st.markdown("""
<div class="title">
✈️ Airline Passenger Forecasting Dashboard
</div>

<div class="subtitle">
Forecast future passenger demand using Deep Learning (LSTM)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------
# Evaluation Metrics
# ---------------------------------
try:
    mae, mse, rmse = Evaluator().evaluate()

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")

except Exception as e:
    st.error(f"Error calculating metrics: {e}")

st.markdown("---")

# ---------------------------------
# Historical Data
# ---------------------------------
left, right = st.columns([2, 1])

with left:

    st.subheader("📈 Historical Passenger Trend")

    fig = px.line(
        df,
        x=df.index,
        y="total_passengers",
        markers=True,
        title="Monthly Passenger Count"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    st.subheader("📋 Recent Data")

    st.dataframe(
        df.tail(12),
        use_container_width=True,
        height=500
    )

st.markdown("---")

# ---------------------------------
# Forecast Section
# ---------------------------------
st.header("🔮 Future Forecast")

if st.button("Generate Forecast"):

    try:

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

        st.success("Forecast Generated Successfully!")

        col1, col2 = st.columns([1, 2])

        with col1:

            st.subheader("Forecast Values")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            csv = forecast_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )

        with col2:

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
                template="plotly_white",
                title="Historical vs Forecast",
                height=500
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")

st.caption("© 2026 Airline Passenger Forecasting Dashboard | Built with Streamlit")
