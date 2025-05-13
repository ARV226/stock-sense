import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pickle
import json

class StockPredictor:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.window_size = 10  # Number of previous days to consider for prediction
        
        # Create directory for saving models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        
        # Load existing models and metrics if available
        self._load_models()
        self._load_metrics()
    
    def _load_models(self):
        """Load trained models if they exist"""
        for filename in os.listdir('models'):
            if filename.endswith('_model.pkl'):
                symbol = filename.split('_')[0]
                model_path = os.path.join('models', f"{symbol}_model.pkl")
                scaler_path = os.path.join('models', f"{symbol}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        # Load the prediction model
                        with open(model_path, 'rb') as f:
                            self.models[symbol] = pickle.load(f)
                        # Load the scaler separately
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
                        print(f"Successfully loaded model and scaler for {symbol}")
                        print(f"Model type: {type(self.models[symbol])}")
                        print(f"Scaler type: {type(self.scalers[symbol])}")
                    except Exception as e:
                        print(f"Error loading model for {symbol}: {e}")
    
    def _load_metrics(self):
        """Load model metrics if they exist"""
        metrics_path = os.path.join('metrics', 'model_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            except Exception as e:
                print(f"Error loading metrics: {e}")
                self.model_metrics = {}
    
    def _save_model(self, symbol, model, scaler):
        """Save model and scaler to disk"""
        try:
            model_path = os.path.join('models', f"{symbol}_model.pkl")
            scaler_path = os.path.join('models', f"{symbol}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        except Exception as e:
            print(f"Error saving model for {symbol}: {e}")
    
    def _save_metrics(self):
        """Save model metrics to disk"""
        try:
            metrics_path = os.path.join('metrics', 'model_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def _prepare_data(self, stock_data):
        """Prepare data for model training and prediction"""
        # Create features from time series data
        for i in range(1, self.window_size + 1):
            for feature in self.feature_columns:
                stock_data[f'{feature}_lag_{i}'] = stock_data[feature].shift(i)
        
        # Drop rows with NaN values
        stock_data = stock_data.dropna()
        
        # Features and target
        X = stock_data.drop(['Close'], axis=1)
        y = stock_data['Close']
        
        return X, y
    
    def _create_model(self, symbol, stock_data, sentiment_score=0):
        """Create and train a new model for the given stock"""
        # Prepare data
        X, y = self._prepare_data(stock_data)
        
        # Add sentiment score as a feature if available
        if sentiment_score != 0:
            X['sentiment'] = sentiment_score
        
        # Create and fit the scaler for feature scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Choose prediction model based on available data size
        if len(X) > 100:
            prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
            print(f"Using RandomForestRegressor as prediction model for {symbol}")
        else:
            prediction_model = LinearRegression()
            print(f"Using LinearRegression as prediction model for {symbol}")
        
        # Train the prediction model
        prediction_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = prediction_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        accuracy = 100 * (1 - (rmse / np.mean(y_test)))
        
        print(f"New model created - Model type: {type(prediction_model)}")
        print(f"New scaler created - Scaler type: {type(scaler)}")
        
        # Store model and metrics separately
        self.models[symbol] = prediction_model  # Store prediction model
        self.scalers[symbol] = scaler  # Store scaler separately
        self.model_metrics[symbol] = {
            'rmse': rmse,
            'accuracy': accuracy,
            'last_updated': datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(X)
        }
        
        # Save model and metrics
        self._save_model(symbol, prediction_model, scaler)
        self._save_metrics()
        
        return prediction_model, scaler, accuracy
    
    def _update_model(self, symbol, stock_data, sentiment_score=0):
        """Update existing model with new data"""
        # Get latest data that might not be included in the model
        try:
            # Parse the last update time and make it timezone-aware in UTC
            last_update = datetime.strptime(self.model_metrics[symbol]['last_updated'], '%Y-%m-%d %H:%M:%S')
            # Make last_update timezone aware (UTC)
            last_update = last_update.replace(tzinfo=pytz.UTC)
            
            # Ensure stock_data index is timezone-aware in UTC
            if not all(hasattr(idx, 'tzinfo') and idx.tzinfo is not None for idx in stock_data.index):
                # If any index is timezone-naive, localize all to UTC
                stock_data_copy = stock_data.copy()
                stock_data_copy.index = stock_data_copy.index.map(lambda dt: 
                                                              dt if hasattr(dt, 'tzinfo') and dt.tzinfo is not None 
                                                              else pd.Timestamp(dt).tz_localize(pytz.UTC))
                new_data = stock_data_copy[stock_data_copy.index > last_update]
            else:
                # All indices are already timezone-aware
                if stock_data.index[0].tzinfo != pytz.UTC:
                    # Convert to UTC if needed
                    stock_data_copy = stock_data.copy()
                    stock_data_copy.index = stock_data_copy.index.map(lambda dt: dt.astimezone(pytz.UTC))
                    new_data = stock_data_copy[stock_data_copy.index > last_update]
                else:
                    # Already in UTC
                    new_data = stock_data[stock_data.index > last_update]
            
            if len(new_data) > 0:
                # If we have new data, create a new model with all data
                print(f"Found {len(new_data)} new data points, updating model")
                model, scaler, accuracy = self._create_model(symbol, stock_data, sentiment_score)
                
                # Debug information
                print(f"Updated model type: {type(model)}")
                print(f"Updated scaler type: {type(scaler)}")
                
                return model, scaler, accuracy
            else:
                # Use existing model
                print(f"No new data, using existing model")
                
                # Verify that we're returning the correct objects
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                accuracy = self.model_metrics[symbol]['accuracy']
                
                # Debug information
                print(f"Existing model type: {type(model)}")
                print(f"Existing scaler type: {type(scaler)}")
                
                # Make sure model is not accidentally a scaler
                if isinstance(model, MinMaxScaler):
                    print("ERROR: Model is a MinMaxScaler! Creating a new model instead.")
                    model, scaler, accuracy = self._create_model(symbol, stock_data, sentiment_score)
                
                return model, scaler, accuracy
        except Exception as e:
            # If there's any error, recreate the model
            print(f"Error updating model, creating new one: {e}")
            model, scaler, accuracy = self._create_model(symbol, stock_data, sentiment_score)
            return model, scaler, accuracy
    
    def predict_stock_price(self, symbol, days_ahead=7, sentiment_score=0):
        """
        Predict stock price for each of the next 7 days
        
        Args:
            symbol: Stock symbol
            days_ahead: Number of days to predict ahead (always uses 7 days regardless of input)
            sentiment_score: Sentiment score from news analysis (-1 to 1)
            
        Returns:
            predicted_prices: List of predicted prices for each of the next 7 days
            confidence: Confidence score of the prediction (0-100)
        """
        # Always predict exactly 7 days ahead, regardless of the input parameter
        days_ahead = 7
        try:
            print(f"Starting prediction for {symbol} with {days_ahead} days ahead")
            
            # Get historical data
            stock_data = self.data_manager.get_stock_data(symbol)
            
            if stock_data is None:
                print(f"No stock data available for {symbol}")
                return None, 0
                
            if len(stock_data) < self.window_size + 1:
                print(f"Insufficient data points for {symbol}: {len(stock_data)} < {self.window_size + 1}")
                return None, 0
            
            print(f"Got data for {symbol} with {len(stock_data)} data points")
            
            # Check if we have a model for this symbol
            if symbol in self.models and symbol in self.scalers:
                print(f"Updating existing model for {symbol}")
                model, scaler, accuracy = self._update_model(symbol, stock_data, sentiment_score)
            else:
                print(f"Creating new model for {symbol}")
                model, scaler, accuracy = self._create_model(symbol, stock_data, sentiment_score)
            
            # Prepare the most recent data point for prediction
            latest_data = stock_data.iloc[-self.window_size:].copy()
            
            # Make iterative predictions for the specified number of days
            current_data = latest_data.copy()
            
            # Prepare to store daily predictions with their dates
            daily_predictions = []
            prediction_dates = []
            
            for i in range(days_ahead):
                # Prepare features
                features = current_data.copy()
                
                # Create lagged features
                for j in range(1, self.window_size + 1):
                    for feature in self.feature_columns:
                        features[f'{feature}_lag_{j}'] = features[feature].shift(j)
                
                # The most recent complete row with all features will be the last row
                features = features.iloc[-1:].drop(['Close'], axis=1)
                
                # Add sentiment score as a feature if available
                if sentiment_score != 0:
                    features['sentiment'] = sentiment_score
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction for this day
                prediction = model.predict(features_scaled)[0]
                
                # Update current_data with the prediction for the next iteration
                new_row = current_data.iloc[-1:].copy()
                
                # Make sure we're using timezone-aware dates consistently
                last_date = new_row.index[0]
                if not hasattr(last_date, 'tzinfo') or last_date.tzinfo is None:
                    # Convert to timezone-aware datetime in UTC
                    new_date = pd.Timestamp(last_date).tz_localize(pytz.UTC) + timedelta(days=1)
                else:
                    # Already timezone-aware, just add days
                    new_date = last_date + timedelta(days=1)
                
                # Store this day's prediction and date
                prediction_dates.append(new_date)
                
                # Apply sentiment adjustment to daily prediction
                sentiment_adjustment = 1 + (sentiment_score * 0.05)  # Up to ±5% adjustment
                adjusted_prediction = prediction * sentiment_adjustment
                daily_predictions.append(adjusted_prediction)
                
                # Update the dataframe for next iteration
                new_row.index = [new_date]
                new_row['Close'] = prediction
                
                # For simplicity, we'll set other values similar to the last known values
                # A more sophisticated approach would predict these values too
                new_row['Open'] = prediction * 0.99  # Slight adjustment for open price
                new_row['High'] = prediction * 1.01  # Slight adjustment for high
                new_row['Low'] = prediction * 0.98   # Slight adjustment for low
                new_row['Volume'] = current_data['Volume'].mean()  # Average volume
                
                # Append to current data
                current_data = pd.concat([current_data, new_row])
                
                # Remove the oldest row to maintain the window size
                current_data = current_data.iloc[1:].copy()
            
            # Adjust confidence based on data quality and sentiment strength
            confidence = min(float(accuracy), 95.0)  # Cap confidence at 95%
            confidence = max(confidence - (abs(sentiment_score) * 10), 30.0)  # Lower confidence with extreme sentiment
            
            # Return all 7 daily predictions with their dates
            prediction_result = list(zip(prediction_dates, daily_predictions))
            
            return prediction_result, confidence
            
        except Exception as e:
            import traceback
            print(f"Error in prediction process: {e}")
            print(traceback.format_exc())
            raise
