"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""

import numpy as np
import joblib

from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor


class Forecaster:

    def __init__(self):

        # Change this if your file is airline-passengers.csv.csv
        self.data_path = r"data/airline-passengers.csv"

        self.model_path = r"models/lstm_model.keras"

        self.scaler_path = r"models/scaler.pkl"

        self.sequence_length = 12

    def forecast(self, future_months=12):

        # --------------------------
        # Step 1 : Load Dataset
        # --------------------------
        loader = DataLoader(self.data_path)
        df = loader.load_data()

        # --------------------------
        # Step 2 : Scale Dataset
        # --------------------------
        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df)

        # --------------------------
        # Step 3 : Load Model
        # --------------------------
        model = load_model(self.model_path)

        # --------------------------
        # Step 4 : Load Scaler
        # --------------------------
        scaler = joblib.load(self.scaler_path)

        # --------------------------
        # Step 5 : Last 12 Months
        # --------------------------
        last_sequence = scaled_df.values[-self.sequence_length:]

        future_predictions = []

        # --------------------------
        # Step 6 : Forecast Loop
        # --------------------------
        for _ in range(future_months):

            input_data = last_sequence.reshape(1, self.sequence_length, 1)

            prediction = model.predict(input_data, verbose=0)

            future_predictions.append(prediction[0][0])

            last_sequence = np.vstack(
                (
                    last_sequence[1:],
                    prediction
                )
            )

        # --------------------------
        # Step 7 : Convert to Original Scale
        # --------------------------
        future_predictions = np.array(future_predictions).reshape(-1, 1)

        future_predictions = scaler.inverse_transform(future_predictions)

        print("\nFuture Forecast Completed.")

        return future_predictions


if __name__ == "__main__":

    forecaster = Forecaster()

    future = forecaster.forecast(future_months=12)

    print("\nNext 12 Month Forecast\n")

    for i, value in enumerate(future, start=1):
        print(f"Month {i}: {value[0]:.2f}")
