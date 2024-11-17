import daal4py as d4p
import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time

def load_xgboost_model(model_path: str) -> XGBRegressor:
    """
    Load the trained XGBoost model from the specified path.
    """
    model = XGBRegressor()
    model.load_model(model_path)
    return model

def convert_model_to_daal4py(xgb_model: XGBRegressor):
    """
    Convert the XGBoost model to a daal4py model for accelerated inference.
    """
    d4p_model = d4p.mb.convert_model(xgb_model)
    return d4p_model

def predict_with_daal4py(d4p_model, input_data: pd.DataFrame) -> np.ndarray:
    """
    Perform prediction using the daal4py accelerated model.
    """
    predictions = d4p_model.predict(input_data)
    return predictions

def main():
    # Load the trained XGBoost model
    xgb_model = load_xgboost_model('xgboost_model.json')
    print("XGBoost model loaded successfully.")

    # Convert the model to daal4py
    d4p_model = convert_model_to_daal4py(xgb_model)
    print("Model converted to daal4py format.")

    # Example input: last 5 market prices (ensure it matches the training feature format)
    recent_prices = np.array([[150.5, 152.3, 151.8, 153.0, 154.2]])
    input_df = pd.DataFrame(recent_prices, columns=[str(i) for i in range(5)])
    # Alternate between daal4py and XGBoost predictions for fair comparison
    n_predictions = 10_000
    daal_times = []
    xgb_times = []
    
    for _ in range(n_predictions):
        # daal4py prediction
        start_time = time.time()
        predict_with_daal4py(d4p_model, input_df)
        daal_times.append(time.time() - start_time)

        # XGBoost prediction
        start_time = time.time()
        xgb_model.predict(input_df)[0]
        xgb_times.append(time.time() - start_time)


    daal_time_mean = np.mean(daal_times)
    daal_time_std = np.std(daal_times)
    print(f"daal4py Prediction Time: {daal_time_mean:.6f} ± {daal_time_std:.6f} seconds")

    xgb_time_mean = np.mean(xgb_times)
    xgb_time_std = np.std(xgb_times)
    print(f"XGBoost Prediction Time: {xgb_time_mean:.6f} ± {xgb_time_std:.6f} seconds")

    print(f"Speedup: {xgb_time_mean / daal_time_mean:.2f}x")

if __name__ == "__main__":
    main() 