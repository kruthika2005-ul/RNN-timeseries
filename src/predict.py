"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""

import joblib
from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit


class Predictor:

    def __init__(self):

        # Change this if your file is airline-passengers.csv.csv
        self.data_path = r"data/airline-passengers.csv"

        self.model_path = r"models/lstm_model.keras"

        self.scaler_path = r"models/scaler.pkl"

    def predict(self):

        # ----------------------------
        # Step 1 : Load Dataset
        # ----------------------------
        loader = DataLoader(self.data_path)
        df = loader.load_data()

        # ----------------------------
        # Step 2 : Preprocess
        # ----------------------------
        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df)

        # ----------------------------
        # Step 3 : Generate Sequences
        # ----------------------------
        generator = SequenceGenerator(sequence_length=12)
        X, y = generator.create_sequences(scaled_df)

        # ----------------------------
        # Step 4 : Train-Test Split
        # ----------------------------
        splitter = TimeSeriesSplit(train_size=0.80)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # ----------------------------
        # Step 5 : Load Model
        # ----------------------------
        model = load_model(self.model_path)
        print("\nModel Loaded Successfully.")

        # ----------------------------
        # Step 6 : Load Scaler
        # ----------------------------
        scaler = joblib.load(self.scaler_path)
        print("Scaler Loaded Successfully.")

        # ----------------------------
        # Step 7 : Predict
        # ----------------------------
        predictions = model.predict(X_test, verbose=0)

        # ----------------------------
        # Step 8 : Inverse Transform
        # ----------------------------
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        print("\nPrediction Completed.")

        return y_test, predictions


if __name__ == "__main__":

    predictor = Predictor()

    actual, predicted = predictor.predict()

    print("\nFirst 10 Predictions\n")

    for i in range(min(10, len(actual))):
        print(
            f"Actual: {actual[i][0]:.2f}    "
            f"Predicted: {predicted[i][0]:.2f}"
        )
