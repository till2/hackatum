import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_and_save_model():
    # Generate synthetic market price data for demonstration
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.uniform(low=100, high=200, size=1000)
    })

    # Feature engineering: using the past 5 prices to predict the next price
    window_size = 5
    for i in range(1, window_size + 1):
        data[f'lag_{i}'] = data['price'].shift(i)

    data.dropna(inplace=True)

    X = data[[f'lag_{i}' for i in range(window_size, 0, -1)]]
    y = data['price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the XGBoost regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Save the trained model
    model.save_model('xgboost_model.json')
    print('Model saved to xgboost_model.json')

if __name__ == "__main__":
    train_and_save_model() 