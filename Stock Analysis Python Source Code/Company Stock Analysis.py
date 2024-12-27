import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch
import numpy as np
from datetime import datetime, timedelta
import json
from langdetect import detect
import pandas as pd
import yfinance as yf
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QMainWindow, QDialog, QTextBrowser
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QEventLoop
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from io import BytesIO
import re
import time
from ta import momentum, trend, volatility
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import atexit
import tempfile
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from ta.volume import VolumeWeightedAveragePrice
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
import signal
import pickle
import traceback
import numpy as np
import pandas as pd
from ta import momentum, trend, volatility
from ta.volume import VolumeWeightedAveragePrice


# Simple DataFetcher for demonstration
class DataFetcher:
    def fetch_historical_data(self, stock_number):
        """Fetches historical stock data using yfinance."""
        ticker = f"{stock_number}.T"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5) # 5 years of data
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No data retrieved for stock {ticker}")
                return pd.DataFrame() # Return empty DataFrame
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error retrieving stock data for {ticker}: {e}")
            return pd.DataFrame()  # Return empty DataFrame

    def scrape_analyst_consensus(self, stock_number):
        # You can use your existing scraping logic here or implement a placeholder
        return {
            'consensus': {'strong_buy': 0, 'buy': 0, 'neutral': 0, 'target_price': None, 'price_potential': None},
            'forecasts': {'revenue': [], 'net_profit': [], 'operating_profit': [], 'ordinary_profit': []}
        }


    def cleanup():
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith('_MEI') or filename.startswith('magi_'):
                try:
                    os.remove(os.path.join(temp_dir, filename))
                except:
                    pass



class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("LSTM training timed out")



class StockUpdateThread(QThread):
    import traceback  # Add this import
    update_signal = pyqtSignal(dict)

    def __init__(self, stock_number, ja_tokenizer, ja_model, en_tokenizer, en_model):
        super().__init__()
        self.stock_number = stock_number
        self.ja_tokenizer = ja_tokenizer
        self.ja_model = ja_model
        self.en_tokenizer = en_tokenizer
        self.en_model = en_model
        self.running = True
        self.predictor = None
        self.model_tracker = ModelTracker()
        self.days_since_last_retrain = 0
        self.data_fetcher = DataFetcher()  # Initialize DataFetcher here

        # Load the model if it exists
        if self.model_tracker.load_model():
            self.predictor = self.model_tracker.best_model
            self.scaler = self.model_tracker.scaler
            self.selected_features = self.model_tracker.selected_features  # Load selected features


    def handle_returns_calculation(self, y_true, y_pred):
        """
        Safely calculate returns and handle edge cases
        
        Parameters:
            y_true: array-like, true values
            y_pred: array-like, predicted values
            
        Returns:
            tuple: (y_true_returns, y_pred_returns) with safely calculated returns
        """
        try:
            EPSILON = 1e-10  # Small constant to avoid division by zero
            
            # Convert inputs to pandas Series to use diff()
            y_true = pd.Series(y_true)
            y_pred = pd.Series(y_pred)
            
            # Calculate returns using log returns to avoid extreme values
            y_true_returns = np.log(y_true + EPSILON).diff()
            y_pred_returns = np.log(y_pred + EPSILON).diff()
            
            # Remove NaN values (first element after diff)
            y_true_returns = y_true_returns.dropna()
            y_pred_returns = y_pred_returns.dropna()
            
            # Remove any infinite values
            mask = np.isfinite(y_true_returns) & np.isfinite(y_pred_returns)
            y_true_returns = y_true_returns[mask]
            y_pred_returns = y_pred_returns[mask]
            
            # Clip extreme values
            y_true_returns = np.clip(y_true_returns, -1, 1)
            y_pred_returns = np.clip(y_pred_returns, -1, 1)
            
            if len(y_true_returns) == 0 or len(y_pred_returns) == 0:
                print("Warning: No valid returns after cleaning")
                return pd.Series([0]), pd.Series([0])  # Return safe default values
                
            return y_true_returns, y_pred_returns
            
        except Exception as e:
            print(f"Error in handle_returns_calculation: {e}")
            traceback.print_exc()
            return pd.Series([0]), pd.Series([0])  # Return safe default values


    def scrape_analyst_consensus(self):
        """
        Scrapes analyst consensus and financial forecast data from Minkabu
        Returns a dictionary containing analyst recommendations and financial forecasts
        """
        url = f"https://minkabu.jp/stock/{self.stock_number}/analyst_consensus"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract analyst consensus
            consensus_text = soup.find('p', string=lambda text: text and '時点における' in text)
            if consensus_text:
                consensus_data = {
                    'strong_buy': 0,
                    'buy': 0,
                    'neutral': 0,
                    'target_price': None,
                    'price_potential': None
                }
                
                text = consensus_text.text
                
                # Extract number of analysts for each recommendation
                strong_buy_match = re.search(r'強気買い(\d+)人', text)
                buy_match = re.search(r'買い(\d+)人', text)
                neutral_match = re.search(r'中立(\d+)人', text)
                
                if strong_buy_match:
                    consensus_data['strong_buy'] = int(strong_buy_match.group(1))
                if buy_match:
                    consensus_data['buy'] = int(buy_match.group(1))
                if neutral_match:
                    consensus_data['neutral'] = int(neutral_match.group(1))
                
                # Extract target price and potential
                target_price_match = re.search(r'平均目標株価は([\d,]+)円', text)
                price_potential_match = re.search(r'株価はあと([\d.]+)％(\w+)', text)
                
                if target_price_match:
                    consensus_data['target_price'] = float(target_price_match.group(1).replace(',', ''))
                if price_potential_match:
                    potential = float(price_potential_match.group(1))
                    direction = 1 if price_potential_match.group(2) == '上昇' else -1
                    consensus_data['price_potential'] = potential * direction
            
            # Extract financial forecasts
            forecasts = {
                'revenue': [],
                'net_profit': [],
                'operating_profit': [],
                'ordinary_profit': []
            }
            
            # Find the table containing financial forecasts
            table = soup.find('table', class_='md_table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) > 0:
                        header = cells[0].text.strip()
                        if header == '売上高':
                            forecasts['revenue'] = [int(cell.text.strip().replace(',', '')) 
                                                for cell in cells[1:] if cell.text.strip() != '---']
                        elif header == '純利益':
                            forecasts['net_profit'] = [int(cell.text.strip().replace(',', '')) 
                                                    for cell in cells[1:] if cell.text.strip() != '---']
                        elif header == '営業利益':
                            forecasts['operating_profit'] = [int(cell.text.strip().replace(',', '')) 
                                                        for cell in cells[1:] if cell.text.strip() != '---']
                        elif header == '経常利益':
                            forecasts['ordinary_profit'] = [int(cell.text.strip().replace(',', '')) 
                                                        for cell in cells[1:] if cell.text.strip() != '---']
            
            return {
                'consensus': consensus_data,
                'forecasts': forecasts
            }
            
        except Exception as e:
            print(f"Error in scrape_analyst_consensus: {e}")
            return None


    def train_ml_model(self):
        try:
            print("Fetching complete historical data...")
            # Use DataFetcher to get data
            df = self.data_fetcher.fetch_historical_data(self.stock_number)

            print(f"Retrieved data spanning {(df.index[-1] - df.index[0]).days / 365.25:.1f} years")

            if df.empty:
                print("No historical data available")
                return None, None

            print(f"Data available from {df.index[0]} to {df.index[-1]}")
            print(f"Total data points: {len(df)}")

            # Advanced feature engineering
            print("Creating enhanced technical features...")

            # Price-based features
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Price_Volatility'] = df['Log_Return'].rolling(window=20).std()
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

            # Volume-based features
            df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_MA30'] = df['Volume'].rolling(window=30).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA10']
            df['VWAP'] = VolumeWeightedAveragePrice(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume']
            ).volume_weighted_average_price()

            # Momentum indicators
            df['RSI'] = momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
            df['Stochastic_K'] = momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

            # Trend indicators
            df['MACD'] = trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = trend.MACD(df['Close']).macd_signal()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['ADX'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            df['DI_plus'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos()
            df['DI_minus'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg()

            # Volatility indicators
            bb = volatility.BollingerBands(df['Close'])
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            df['ATR'] = volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

            # Moving averages and crossovers
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

            # Moving average crossover signals
            df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)

            # Market breadth indicators
            print("Adding market indicators...")
            try:
                # Get data using df's date range
                start_date = df.index[0]
                end_date = df.index[-1]

                # Nikkei 225 data
                nikkei = yf.download("^N225", start=start_date, end=end_date)
                df['Nikkei_Return'] = nikkei['Close'].pct_change()
                df['Nikkei_Volatility'] = nikkei['Close'].rolling(window=20).std()
                df['Beta'] = df['Log_Return'].rolling(window=60).cov(df['Nikkei_Return']) / df['Nikkei_Return'].rolling(window=60).var()

                # USD/JPY exchange rate
                usdjpy = yf.download("USDJPY=X", start=start_date, end=end_date)
                df['USDJPY_Return'] = usdjpy['Close'].pct_change()
                df['USDJPY_Volatility'] = usdjpy['Close'].rolling(window=20).std()

                # TOPIX index
                topix = yf.download("1605.T", start=start_date, end=end_date)
                df['TOPIX_Return'] = topix['Close'].pct_change()
                df['TOPIX_Volatility'] = topix['Close'].rolling(window=20).std()
                df['TOPIX_Correlation'] = df['Log_Return'].rolling(window=60).corr(topix['Close'].pct_change())

                # Interest rate proxy
                jgb = yf.download("^TNX", start=start_date, end=end_date)
                df['Interest_Rate_Change'] = jgb['Close'].pct_change()
                df['Interest_Rate_MA'] = jgb['Close'].rolling(window=20).mean()

            except Exception as e:
                print(f"Error fetching market data: {e}")
                for col in ['Nikkei_Return', 'Nikkei_Volatility', 'Beta',
                        'USDJPY_Return', 'USDJPY_Volatility',
                        'TOPIX_Return', 'TOPIX_Volatility', 'TOPIX_Correlation',
                        'Interest_Rate_Change', 'Interest_Rate_MA']:
                    df[col] = 0.0

            # Add analyst consensus data
            print("Fetching analyst consensus data...")
            analyst_data = self.data_fetcher.scrape_analyst_consensus(self.stock_number)
            if analyst_data and analyst_data['consensus']:
                consensus = analyst_data['consensus']
                total_analysts = consensus['strong_buy'] + consensus['buy'] + consensus['neutral']

                # Analyst recommendations
                if total_analysts > 0:
                    df['Analyst_Buy_Ratio'] = (consensus['strong_buy'] + consensus['buy']) / total_analysts
                    df['Analyst_Strong_Buy_Ratio'] = consensus['strong_buy'] / total_analysts
                    df['Analyst_Neutral_Ratio'] = consensus['neutral'] / total_analysts

                # Target price analysis
                if consensus['target_price'] is not None:
                    df['Price_To_Target_Ratio'] = df['Close'] / consensus['target_price']
                    df['Target_Price_Premium'] = (consensus['target_price'] - df['Close']) / df['Close']

                # Price potential
                if consensus['price_potential'] is not None:
                    df['Analyst_Price_Potential'] = consensus['price_potential']
                    df['Price_Potential_Signal'] = (consensus['price_potential'] > 0).astype(int)

            # Add forecast features
            print("Processing financial forecasts...")
            if analyst_data and analyst_data['forecasts']:
                forecasts = analyst_data['forecasts']

                # Revenue growth metrics
                if forecasts['revenue'] and len(forecasts['revenue']) >= 2:
                    revenue_growth = ((forecasts['revenue'][-1] - forecasts['revenue'][-2]) /
                                    forecasts['revenue'][-2])
                    df['Expected_Revenue_Growth'] = revenue_growth
                    df['Revenue_Growth_Signal'] = (revenue_growth > 0).astype(int)

                # Profit growth metrics
                if forecasts['net_profit'] and len(forecasts['net_profit']) >= 2:
                    profit_growth = ((forecasts['net_profit'][-1] - forecasts['net_profit'][-2]) /
                                    forecasts['net_profit'][-2])
                    df['Expected_Profit_Growth'] = profit_growth
                    df['Profit_Growth_Signal'] = (profit_growth > 0).astype(int)

                # Operating profit trends
                if forecasts['operating_profit'] and len(forecasts['operating_profit']) >= 2:
                    op_profit_growth = ((forecasts['operating_profit'][-1] - forecasts['operating_profit'][-2]) /
                                        forecasts['operating_profit'][-2])
                    df['Expected_Operating_Profit_Growth'] = op_profit_growth

                # Ordinary profit trends
                if forecasts['ordinary_profit'] and len(forecasts['ordinary_profit']) >= 2:
                    ord_profit_growth = ((forecasts['ordinary_profit'][-1] - forecasts['ordinary_profit'][-2]) /
                                        forecasts['ordinary_profit'][-2])
                    df['Expected_Ordinary_Profit_Growth'] = ord_profit_growth

            # Target variables with multiple horizons
            for horizon in [1, 3, 5, 10]:
                df[f'Target_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1

            # Drop rows with NaN values
            df.dropna(inplace=True)

            if len(df) < 60:  # Minimum required data points
                print("Insufficient data for training")
                return None, None

            # Feature selection
            feature_cols = [col for col in df.columns if col not in ['Target_1d', 'Target_3d', 'Target_5d', 'Target_10d']]
            X = df[feature_cols]
            y = df['Target_1d']  # Primary target is 1-day ahead prediction

            # Advanced feature selection using RFE
            rfe = RFE(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                n_features_to_select=min(50, len(feature_cols))
            )
            X_selected = rfe.fit_transform(X, y)
            self.selected_features = [feature for feature, selected in zip(feature_cols, rfe.support_) if selected]
            print(f"Selected {len(self.selected_features)} features: {self.selected_features}")

            # Train-test split with time-based validation
            train_size = int(len(df) * 0.8)
            X_train = X_selected[:train_size]
            X_test = X_selected[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]

            # Feature scaling - Store scaler as an instance variable
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Create enhanced ensemble model
            models = [
                ('rf', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=42
                )),
                ('gb', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )),
                ('xgb', xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    random_state=42
                ))
            ]

            # Create and train ensemble
            ensemble = VotingRegressor(estimators=models)
            ensemble.fit(X_train_scaled, y_train)

            try:
                # Make predictions
                y_pred = ensemble.predict(X_test_scaled)
                
                # Convert to numpy arrays for easier manipulation
                y_test_values = np.array(y_test)
                y_pred_values = np.array(y_pred)
                
                # Calculate directional accuracy using numpy arrays
                y_test_diff = np.diff(y_test_values)
                y_pred_diff = np.diff(y_pred_values)
                
                # Calculate directional accuracy (whether the signs match)
                matching_directions = (np.sign(y_test_diff) == np.sign(y_pred_diff))
                direction_accuracy = np.mean(matching_directions)
                
                # Calculate MAPE and handle zero values
                non_zero_mask = y_test_values != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_test_values[non_zero_mask] - y_pred_values[non_zero_mask]) 
                                        / y_test_values[non_zero_mask]))
                else:
                    mape = np.mean(np.abs(y_test_values - y_pred_values))  # Fallback when all values are zero
                
                # Calculate RMSE and normalize it
                rmse = np.sqrt(mean_squared_error(y_test_values, y_pred_values))
                price_range = np.ptp(y_test_values)  # Peak to peak (max - min)
                if price_range > 0:
                    normalized_rmse = rmse / price_range
                else:
                    normalized_rmse = rmse
                
                # Calculate final accuracy score
                direction_component = direction_accuracy
                mape_component = np.clip(1 - mape, 0, 1)  # Convert MAPE to accuracy and clip
                rmse_component = np.clip(1 - normalized_rmse, 0, 1)  # Convert RMSE to accuracy and clip
                
                # Combine components with weights
                accuracy = (0.5 * direction_component +
                        0.3 * mape_component +
                        0.2 * rmse_component)
                
                # Ensure final accuracy is properly bounded
                accuracy = np.clip(accuracy, 0, 1)
                
                print(f"\nDetailed Prediction Analysis:")
                print(f"Direction Accuracy: {direction_accuracy:.4f}")
                print(f"MAPE Score: {mape_component:.4f}")
                print(f"RMSE Score: {rmse_component:.4f}")
                print(f"Final Combined Accuracy: {accuracy:.4f}")
                
                # Save model if improved
                if accuracy > 0:
                    print(f"Saving model with accuracy: {accuracy}")
                    self.model_tracker.save_model(ensemble, accuracy, self.scaler, self.selected_features)
                
                    # Compare with existing best model
                    if self.model_tracker.best_model is not None:
                        if accuracy < self.model_tracker.best_accuracy:
                            print(f"Using previous best model with accuracy: {self.model_tracker.best_accuracy}")
                            return self.model_tracker.best_model, self.model_tracker.best_accuracy
                
                # Assign the trained ensemble and return
                self.predictor = ensemble
                return ensemble, accuracy

            except Exception as e:
                print(f"Error in accuracy calculation: {e}")
                traceback.print_exc()
                self.predictor = ensemble  # Still assign the ensemble even if accuracy calculation fails
                return ensemble, 0.0  # Return 0.0 accuracy on error

        except Exception as e:
            print(f"Error in train_ml_model: {e}")
            traceback.print_exc()
            return None, None

    def prepare_prediction_data(self, dates, prices, current_price):
        print("Preparing prediction data...")
        try:
            # Create DataFrame with latest data first
            df = pd.DataFrame({'Date': dates, 'Close': prices})
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index(ascending=False)  # Most recent first

            # Get historical data for better feature calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=250)  # Get more data for SMA200

            try:
                hist_data = yf.download(f"{self.stock_number}.T", start=start_date, end=end_date)
                if not hist_data.empty:
                    # Merge historical data with current data
                    df = pd.concat([df, hist_data[~hist_data.index.isin(df.index)]])
                    df = df.sort_index(ascending=False)
            except Exception as e:
                print(f"Error fetching historical data: {e}")

            # Ensure data completeness
            required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Open':
                        df[col] = df['Close'].shift(-1)
                    elif col in ['High', 'Low']:
                        df[col] = df['Close']
                    elif col == 'Adj Close':
                        df[col] = df['Close']
                    elif col == 'Volume':
                        df[col] = df['Close'].rolling(window=5).mean() * 1000  # Approximate volume

            # Calculate all features
            df = self.calculate_features(df)

            # Add analyst consensus data if available
            analyst_data = self.scrape_analyst_consensus()
            if analyst_data and analyst_data['consensus']:
                consensus = analyst_data['consensus']
                total_analysts = consensus['strong_buy'] + consensus['buy'] + consensus['neutral']
                
                if total_analysts > 0:
                    df['Analyst_Buy_Ratio'] = (consensus['strong_buy'] + consensus['buy']) / total_analysts
                    df['Analyst_Strong_Buy_Ratio'] = consensus['strong_buy'] / total_analysts
                    df['Analyst_Neutral_Ratio'] = consensus['neutral'] / total_analysts
                
                if consensus['target_price'] is not None:
                    df['Price_To_Target_Ratio'] = df['Close'] / consensus['target_price']
                    df['Target_Price_Premium'] = (consensus['target_price'] - df['Close']) / df['Close']
                
                if consensus['price_potential'] is not None:
                    df['Analyst_Price_Potential'] = consensus['price_potential']
                    df['Price_Potential_Signal'] = (consensus['price_potential'] > 0).astype(int)

            # Add forecast features
            if analyst_data and analyst_data['forecasts']:
                forecasts = analyst_data['forecasts']
                
                # Revenue growth metrics
                if forecasts['revenue'] and len(forecasts['revenue']) >= 2:
                    revenue_growth = ((forecasts['revenue'][-1] - forecasts['revenue'][-2]) /
                                    forecasts['revenue'][-2])
                    df['Expected_Revenue_Growth'] = revenue_growth
                    df['Revenue_Growth_Signal'] = (revenue_growth > 0).astype(int)

                # Profit metrics
                if forecasts['net_profit'] and len(forecasts['net_profit']) >= 2:
                    profit_growth = ((forecasts['net_profit'][-1] - forecasts['net_profit'][-2]) /
                                    forecasts['net_profit'][-2])
                    df['Expected_Profit_Growth'] = profit_growth
                    df['Profit_Growth_Signal'] = (profit_growth > 0).astype(int)

            # Use the most recent data point
            latest_data = df.iloc[0:1].copy()
            if current_price is not None:
                latest_data.loc[latest_data.index[0], 'Close'] = current_price

            # Handle missing features
            if hasattr(self, 'selected_features'):
                missing_features = set(self.selected_features) - set(latest_data.columns)
                if missing_features:
                    print(f"Adding missing features: {missing_features}")
                    for feature in missing_features:
                        latest_data[feature] = 0.0

                # Select only the features used during training
                latest_data = latest_data[self.selected_features]
                print(f"Final features for prediction: {latest_data.columns.tolist()}")

            # Fill any remaining NaN values
            latest_data = latest_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Verify no NaN values remain
            if latest_data.isnull().values.any():
                print("Warning: NaN values still present after cleaning")
                latest_data = latest_data.fillna(0)

            return latest_data

        except Exception as e:
            print(f"Error in prepare_prediction_data: {e}")
            traceback.print_exc()
            return None

    def calculate_features(self, df):
        """Calculate all technical indicators and features"""
        try:
            # Price-based features
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Price_Volatility'] = df['Log_Return'].rolling(window=20).std()
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

            # Volume features
            df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_MA30'] = df['Volume'].rolling(window=30).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA10']
            df['VWAP'] = VolumeWeightedAveragePrice(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume']
            ).volume_weighted_average_price()

            # Technical indicators
            df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
            df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
            df['Stochastic_K'] = momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
            df['MACD'] = trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = trend.MACD(df['Close']).macd_signal()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['ADX'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            df['DI_plus'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos()
            df['DI_minus'] = trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg()

            # Bollinger Bands
            bb = volatility.BollingerBands(df['Close'])
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            df['ATR'] = volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

            # Market indicators
            try:
                end_date = df.index[0]
                start_date = end_date - timedelta(days=30)
                
                # Fetch market data
                nikkei = yf.download("^N225", start=start_date, end=end_date)
                usdjpy = yf.download("USDJPY=X", start=start_date, end=end_date)
                topix = yf.download("1605.T", start=start_date, end=end_date)
                jgb = yf.download("^TNX", start=start_date, end=end_date)

                # Calculate market features
                df['Nikkei_Return'] = nikkei['Close'].pct_change().fillna(0)
                df['Nikkei_Volatility'] = nikkei['Close'].rolling(window=20).std().fillna(0)
                df['Beta'] = df['Log_Return'].rolling(window=60).cov(nikkei['Close'].pct_change()).fillna(0)
                df['USDJPY_Return'] = usdjpy['Close'].pct_change().fillna(0)
                df['USDJPY_Volatility'] = usdjpy['Close'].rolling(window=20).std().fillna(0)
                df['TOPIX_Return'] = topix['Close'].pct_change().fillna(0)
                df['TOPIX_Volatility'] = topix['Close'].rolling(window=20).std().fillna(0)
                df['TOPIX_Correlation'] = df['Log_Return'].rolling(window=60).corr(topix['Close'].pct_change()).fillna(0)
                df['Interest_Rate_Change'] = jgb['Close'].pct_change().fillna(0)
                df['Interest_Rate_MA'] = jgb['Close'].rolling(window=20).mean().fillna(0)

            except Exception as e:
                print(f"Error calculating market indicators: {e}")
                market_columns = ['Nikkei_Return', 'Nikkei_Volatility', 'Beta', 'USDJPY_Return',
                                'USDJPY_Volatility', 'TOPIX_Return', 'TOPIX_Volatility',
                                'TOPIX_Correlation', 'Interest_Rate_Change', 'Interest_Rate_MA']
                for col in market_columns:
                    df[col] = 0.0

            return df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        except Exception as e:
            print(f"Error in calculate_features: {e}")
            return df


    def predict_next_day_price(self, latest_data):
        try:
            if self.predictor is None:
                print("No trained model available for prediction")
                return None
            
            if not hasattr(self, 'scaler') or self.scaler is None:
                print("No scaler available for prediction")
                return None

            print("Making prediction with latest data...")
            
            # Verify data integrity
            if latest_data is None or latest_data.empty:
                print("No valid data for prediction")
                return None

            # Ensure feature consistency
            if set(latest_data.columns) != set(self.selected_features):
                missing_features = set(self.selected_features) - set(latest_data.columns)
                extra_features = set(latest_data.columns) - set(self.selected_features)
                print(f"Feature mismatch. Missing: {missing_features}, Extra: {extra_features}")
                return None

            # Final data cleaning
            X = latest_data[self.selected_features]
            X = X.fillna(0)  # Replace any remaining NaN values with 0

            # Scale the features
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                print(f"Error in feature scaling: {e}")
                return None

            # Make prediction
            try:
                prediction = self.predictor.predict(X_scaled)[0]
                current_price = latest_data['Close'].iloc[0]

                # Calculate percentage change
                pct_change = ((prediction - current_price) / current_price) * 100

                # Limit maximum daily change
                MAX_DAILY_CHANGE = 7.0
                if abs(pct_change) > MAX_DAILY_CHANGE:
                    print(f"Warning: Prediction {pct_change:.2f}% exceeded maximum daily change of ±{MAX_DAILY_CHANGE}%")
                    if pct_change > 0:
                        prediction = current_price * (1 + MAX_DAILY_CHANGE/100)
                    else:
                        prediction = current_price * (1 - MAX_DAILY_CHANGE/100)
                    print(f"Adjusted prediction to {prediction:.2f} ({MAX_DAILY_CHANGE if pct_change > 0 else -MAX_DAILY_CHANGE:.2f}%)")

                # Track the prediction
                self.model_tracker.track_performance(current_price, prediction)

                print(f"Prediction successful: {prediction:.2f} (change: {pct_change:.2f}%)")
                return prediction

            except Exception as e:
                print(f"Error making prediction: {e}")
                traceback.print_exc()
                return None

        except Exception as e:
            print(f"Error in predict_next_day_price: {e}")
            traceback.print_exc()
            return None


    def run(self):
        while self.running:
            try:
                print("Fetching latest data...")
                data = self.fetch_latest_data()
                
                current_price = data['current_price']
                stock_data = data['stock_data']
                
                # Train model if not already 
                if self.predictor is None:
                    print("Training model...")
                    self.predictor, accuracy = self.train_ml_model()
                    if accuracy is not None:
                        self.current_accuracy = accuracy
                        data['model_accuracy'] = accuracy
                    else:
                        data['model_accuracy'] = None
                
                # Make prediction if possible
                if self.predictor and stock_data and len(stock_data) > 0:
                    try:
                        dates, prices = zip(*stock_data)
                        latest_data = self.prepare_prediction_data(dates, prices, current_price)
                        
                        if latest_data is not None:
                            next_day_prediction = self.predict_next_day_price(latest_data)
                            print(f"Prediction result: {next_day_prediction}")
                            data['next_day_prediction'] = next_day_prediction
                        else:
                            print("Failed to prepare prediction data")
                            data['next_day_prediction'] = None
                    except Exception as e:
                        print(f"Error in prediction process: {e}")
                        data['next_day_prediction'] = None
                else:
                    print("Cannot make prediction: predictor or stock data not available")
                    data['next_day_prediction'] = None
                
                print("Emitting update signal...")
                self.update_signal.emit(data)
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in run loop: {e}")
                time.sleep(1)


    def fetch_latest_data(self):
        current_price = self.get_current_stock_price()
        company_name = self.get_company_name()
        nikkei_news = self.scrape_nikkei_news()
        yahoo_news = self.scrape_yahoo_finance_news()
        nikkei_sentiment = self.analyze_sentiment(nikkei_news)
        yahoo_sentiment = self.analyze_sentiment(yahoo_news)
        stock_data = self.get_stock_data()
        psr, pbr = self.scrape_psr_pbr()
        roa, roe = self.scrape_roa_roe()
        analyst_data = self.scrape_analyst_consensus()
        
        # Store model accuracy if we have a trained model
        model_accuracy = None
        if hasattr(self, 'current_accuracy'):
            model_accuracy = self.current_accuracy
        
        return {
            'current_price': current_price,
            'company_name': company_name,
            'nikkei_sentiment': nikkei_sentiment,
            'yahoo_sentiment': yahoo_sentiment,
            'nikkei_news': nikkei_news,
            'yahoo_news': yahoo_news,
            'stock_data': stock_data,
            'psr': psr,
            'pbr': pbr,
            'roa': roa,
            'roe': roe,
            'model_accuracy': model_accuracy,
            'analyst_data': analyst_data
        }


    def get_current_stock_price(self):
        url = f"https://finance.yahoo.co.jp/quote/{self.stock_number}.T"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        price_element = soup.select_one('span._3rXWJKZF')
        if price_element:
            price_text = price_element.text.strip().replace(',', '')
            try:
                return float(price_text)
            except ValueError:
                return None
        return None


    def scrape_nikkei_news(self):
        url = f"https://www.nikkei.com/nkd/company/news/?scode={self.stock_number}&ba=1"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.find_all('a', href=lambda href: href and "/nkd/company/article/" in href)
        news_data = []
        for item in news_items[:10]:  # Get latest 10 news items
            title = item.text.strip()
            url = "https://www.nikkei.com" + item['href']
            news_data.append({"title": title, "url": url})
        return news_data


    def scrape_yahoo_finance_news(self):
        url = f"https://finance.yahoo.co.jp/quote/{self.stock_number}.T/news"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.find_all('a', href=lambda href: href and "/news/" in href)
        news_data = []
        for item in news_items[:10]:  # Get latest 10 news items
            title = item.text.strip()
            article_url = item['href']
            if not article_url.startswith('http'):
                article_url = "https://finance.yahoo.co.jp" + article_url
            news_data.append({"title": title, "url": article_url})
        return news_data


    def analyze_sentiment(self, news_data):
        sentiments = []
        for news in news_data:
            try:
                lang = detect(news['title'])
            except:
                lang = 'ja'  # Default to Japanese if detection fails

            if lang == 'ja':
                tokenizer = self.ja_tokenizer
                model = self.ja_model
            else:
                tokenizer = self.en_tokenizer
                model = self.en_model

            inputs = tokenizer(news['title'], return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            sentiment_score = torch.softmax(outputs.logits, dim=1).tolist()[0]
            sentiments.append(sentiment_score[0])

        return sum(sentiments) / len(sentiments) if sentiments else None


    def get_company_name(self):
        url = f"https://finance.yahoo.co.jp/quote/{self.stock_number}.T"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.strip()
            company_name = title.split('【')[0].strip()
            return company_name
        else:
            return "Company name not found"


    def get_stock_data(self):
        ticker = f"{self.stock_number}.T"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Increased from 30 to 60 days
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No data retrieved for stock {ticker}")
                return None
            
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            stock_data = [(row['Date'], row['Close']) for _, row in df.iterrows()]
            
            stock_data.sort(key=lambda x: x[0], reverse=True)
            print(f"Retrieved {len(stock_data)} data points for stock {ticker}")
            return stock_data[:60]  # Return up to 60 days of data
        except Exception as e:
            print(f"Error retrieving stock data for {ticker}: {e}")
            return None


    def scrape_psr_pbr(self):
        url = f"https://minkabu.jp/stock/{self.stock_number}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            psr = pbr = None
            
            rows = soup.find_all('tr', class_='ly_vamd')
            
            for row in rows:
                th = row.find('th', class_='ly_vamd_inner ly_colsize_3_fix tal wsnw')
                if th:
                    if th.text.strip() == 'PSR':
                        td = row.find('td', class_='ly_vamd_inner ly_colsize_9_fix fwb tar wsnw')
                        if td:
                            psr = float(td.text.strip().replace('倍', ''))
                    elif th.text.strip() == 'PBR':
                        td = row.find('td', class_='ly_vamd_inner ly_colsize_9_fix fwb tar wsnw')
                        if td:
                            pbr = float(td.text.strip().replace('倍', ''))
            
            return psr, pbr
        except Exception as e:
            print(f"Error fetching or parsing PSR/PBR data: {e}")
            return None, None


    def scrape_roa_roe(self):
        url = f"https://minkabu.jp/stock/{self.stock_number}/settlement"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            chart_data = soup.find('settlement-chart-profitability')
            if chart_data:
                chart_elements = chart_data.get(':chart-elements')
                if chart_elements:
                    chart_json = json.loads(chart_elements.replace('&quot;', '"'))
                    
                    roa = chart_json['roa'][-1] if 'roa' in chart_json and chart_json['roa'] else None
                    roe = chart_json['roe'][-1] if 'roe' in chart_json and chart_json['roe'] else None
                    
                    print(f"Scraped ROA: {roa}, ROE: {roe}")
                    return roa, roe
            
            print("ROA and ROE data not found in chart elements")
            return None, None
        except Exception as e:
            print(f"Error in scrape_roa_roe: {e}")
            return None, None


    def stop(self):
        self.running = False



class SentimentPopup(QDialog):
    def __init__(self, source, sentiment, news_data):
        super().__init__()
        self.setWindowTitle(f"{source} Sentiment Analysis")
        self.setGeometry(200, 200, 400, 300)
        self.setStyleSheet("background-color: #001a1a; color: #00ff00;")

        layout = QVBoxLayout()

        sentiment_label = QLabel(f"Overall Sentiment: {sentiment}")
        sentiment_label.setStyleSheet("font-size: 16px; color: #ff8c00;")
        layout.addWidget(sentiment_label)

        news_browser = QTextBrowser()
        news_browser.setStyleSheet("background-color: #002a2a; border: none;")
        for item in news_data:
            news_browser.append(f"Title: {item['title']}")
            news_browser.append(f"URL: <a href='{item['url']}'>{item['url']}</a>")
            news_browser.append("\n")
        layout.addWidget(news_browser)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet("""
            background-color: #001a1a;
            color: #00ff00;
            border: 2px solid #ff8c00;
            border-radius: 5px;
            padding: 5px;
        """)
        layout.addWidget(close_button)

        self.setLayout(layout)



class ImprovedStockPredictor:
    def __init__(self):
        self.models = [
            ('RF', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('GB', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('LSTM', None)  # LSTM model will be created during training
        ]
        self.ensemble = None
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = None
        self.scaler = StandardScaler()

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def preprocess_data(self, X, y=None):
        X_imputed = self.imputer.fit_transform(X)
        if y is not None:
            if self.feature_selector is None:
                self.feature_selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=min(10, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X_imputed, y)
            X_scaled = self.scaler.fit_transform(X_selected)
        else:
            if self.feature_selector is None:
                return X_imputed, y
            X_selected = self.feature_selector.transform(X_imputed)
            X_scaled = self.scaler.transform(X_selected)
        return X_scaled, y

    def prepare_lstm_data(self, X, y, lookback):
        X_lstm, y_lstm = [], []
        for i in range(len(X) - lookback):
            X_lstm.append(X[i:(i + lookback)])
            y_lstm.append(y.iloc[i + lookback])
        return np.array(X_lstm), np.array(y_lstm)

    def train(self, X, y):
        X_processed, y_processed = self.preprocess_data(X, y)
        
        tscv = TimeSeriesSplit(n_splits=5)
        for name, model in self.models:
            scores = []
            for train_index, test_index in tscv.split(X_processed):
                X_train, X_test = X_processed[train_index], X_processed[test_index]
                y_train, y_test = y_processed.iloc[train_index], y_processed.iloc[test_index]
                
                if name == 'LSTM':
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # Set a 5-minute timeout

                        lookback = 30  # Adjust as needed
                        X_train_lstm, y_train_lstm = self.prepare_lstm_data(X_train, y_train, lookback)
                        X_test_lstm, y_test_lstm = self.prepare_lstm_data(X_test, y_test, lookback)
                        
                        model = self.create_lstm_model((lookback, X_train.shape[1]))
                        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                        model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, 
                                validation_split=0.2, callbacks=[early_stopping], verbose=0)
                        
                        y_pred = model.predict(X_test_lstm)
                        score = -mean_squared_error(y_test_lstm, y_pred)  # Negative MSE for consistency with sklearn

                        signal.alarm(0)  # Cancel the alarm if training completes successfully
                    except TimeoutException:
                        print("LSTM training timed out. Skipping LSTM model.")
                        score = float('-inf')
                    except Exception as e:
                        print(f"Error in LSTM training: {e}")
                        score = float('-inf')
                else:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                
                scores.append(score)
            
            print(f"{name} average score: {np.mean(scores)}")
            
            if name == 'LSTM':
                self.models[2] = (name, model)  # Update LSTM model in the list
        # Create a simple ensemble by averaging predictions
        self.ensemble = self.models

    def predict(self, X):
        X_processed, _ = self.preprocess_data(X)
        predictions = []
        
        for name, model in self.ensemble:
            if name == 'LSTM':
                lookback = 30  # Should match the lookback used in training
                X_lstm, _ = self.prepare_lstm_data(X_processed, np.zeros(len(X_processed)), lookback)
                pred = model.predict(X_lstm)
                predictions.append(pred.flatten())
            else:
                predictions.append(model.predict(X_processed))
        
        return np.mean(predictions, axis=0)


class MAGIStockAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Data storage
        self.current_stock_data = None
        self.nikkei_news_data = []
        self.yahoo_news_data = []
        self.previous_results = {
            'casper': '',
            'balthasar': '',
            'melchior': ''
        }

        # Timer for graph updates
        self.graph_update_timer = QTimer(self)
        self.graph_update_timer.timeout.connect(self.update_graph)
        self.graph_update_timer.start(1000)  # Update every second

        # Determine the running environment
        if getattr(sys, 'frozen', False):
            bundle_dir = sys._MEIPASS
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize tokenizers and models
        self.ja_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.ja_model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")
        self.en_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.en_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        # Other initializations
        self.flicker_timers = {}
        self.update_thread = None

        # Set window properties
        self.showFullScreen()
        self.setWindowState(Qt.WindowState.WindowFullScreen)


    def open_sentiment_popup(self, url):
        source = url.toString()
        if source == 'nikkei':
            sentiment = self.sentiment_to_text(self.nikkei_sentiment)
            news_data = self.nikkei_news_data
        elif source == 'yahoo':
            sentiment = self.sentiment_to_text(self.yahoo_sentiment)
            news_data = self.yahoo_news_data
        else:
            return

        popup = SentimentPopup(source.capitalize(), sentiment, news_data)
        popup.exec()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.showNormal()
        super().keyPressEvent(event)


    def analyze_stock(self):
        print("Analyze stock method called")
        stock_number = self.stock_input.text()
        purchase_price = self.price_input.text()
        if purchase_price.lower() == 'n/a' or purchase_price == '':
            purchase_price = None
        else:
            try:
                purchase_price = float(purchase_price)
            except ValueError:
                self.show_error("Invalid purchase price. Using N/A.")
                purchase_price = None

        print(f"Stock number: {stock_number}, Purchase price: {purchase_price}")

        # Start flickering
        for component in [self.casper, self.balthasar, self.melchior]:
            self.start_flicker(component)
            component.findChild(QTextBrowser).setText("Loading...")

        # Stop existing update thread if running
        if self.update_thread and self.update_thread.isRunning():
            print("Stopping existing update thread")
            self.update_thread.stop()
            self.update_thread.wait()

        # Start new update thread
        print("Creating new update thread")
        self.update_thread = StockUpdateThread(stock_number, self.ja_tokenizer, self.ja_model, self.en_tokenizer, self.en_model)
        print("Connecting update signal...")
        self.update_thread.update_signal.connect(lambda data: self.update_display(data, purchase_price))
        print("Starting update thread...")
        self.update_thread.start()


    def has_content_changed(self, component_name, new_content):
        if self.previous_results[component_name] != new_content:
            self.previous_results[component_name] = new_content
            return True
        return False


    def update_graph(self):
        if self.current_stock_data:
            component = self.balthasar  # Assuming we're updating BALTHASAR's graph
            dates = [date for date, _ in self.current_stock_data]
            prices = [price for _, price in self.current_stock_data]

            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#001a1a')
            ax.plot(dates, prices, color='#00ff00')
            ax.set_facecolor('#001a1a')
            ax.tick_params(axis='x', colors='#00ff00')
            ax.tick_params(axis='y', colors='#00ff00')
            ax.spines['bottom'].set_color('#00ff00')
            ax.spines['top'].set_color('#00ff00')
            ax.spines['left'].set_color('#00ff00')
            ax.spines['right'].set_color('#00ff00')
            plt.title('Live Stock Pattern', color='#00ff00')
            plt.xticks(rotation=45)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            
            graph_label = component.findChild(QLabel)
            graph_label.setPixmap(pixmap)
            
            plt.close(fig)


    def initUI(self):
        self.setWindowTitle('MAGI Stock Analysis System')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #000000; color: #00ff00;")
        self.nikkei_sentiment = None
        self.yahoo_sentiment = None
        self.nikkei_news_data = []
        self.yahoo_news_data = []

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel('MAGI Stock Analysis System')
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 24px; color: #ff8c00; margin-bottom: 10px;")
        main_layout.addWidget(header)

        # Input section
        input_layout = QHBoxLayout()
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter stock number")
        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("Purchase price (optional)")
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze_stock)

        # Style for input fields
        input_style = """
            QLineEdit {
                background-color: #001a1a;
                color: #00ff00;
                border: 1px solid #00ff00;
                border-radius: 5px;
                padding: 5px;
            }
        """
        self.stock_input.setStyleSheet(input_style)
        self.price_input.setStyleSheet(input_style)

        # Specific style for Analyze button with orange rectangle
        analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #001a1a;
                color: #00ff00;
                border: 2px solid #ff8c00;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #002a2a;
            }
        """)
        
        input_layout.addWidget(self.stock_input)
        input_layout.addWidget(self.price_input)
        input_layout.addWidget(analyze_button)
        main_layout.addLayout(input_layout)

        # MAGI components
        magi_layout = QHBoxLayout()
        magi_layout.setSpacing(10)
        self.melchior = self.create_magi_component("MELCHIOR • 1")
        self.balthasar = self.create_magi_component("BALTHASAR • 2")
        self.casper = self.create_magi_component("CASPER • 3")
        magi_layout.addWidget(self.melchior)
        magi_layout.addWidget(self.balthasar)
        magi_layout.addWidget(self.casper)
        main_layout.addLayout(magi_layout, stretch=1)

        self.setLayout(main_layout)


    def create_magi_component(self, title):
        component = QWidget()
        component.setStyleSheet("""
            background-color: #001a1a;
            border: 2px solid #ff8c00;
            border-radius: 5px;
        """)
        layout = QVBoxLayout(component)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; color: #ff8c00; background-color: transparent;")
        
        content = QTextBrowser()
        content.setOpenExternalLinks(False)
        content.setStyleSheet("border: none; background-color: transparent; color: #00ff00;")
        content.anchorClicked.connect(self.open_sentiment_popup)
        
        graph_label = QLabel()
        graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_label.setStyleSheet("background-color: transparent;")
        
        layout.addWidget(title_label)
        layout.addWidget(content)
        layout.addWidget(graph_label)
        
        return component


    def show_detailed_explanation(self, explanation):
        detail_popup = QDialog(self)
        detail_popup.setWindowTitle("Detailed Recommendation Explanation")
        detail_popup.setStyleSheet("background-color: #001a1a; color: #00ff00;")
        
        layout = QVBoxLayout()
        
        text_browser = QTextBrowser()
        text_browser.setHtml(f"<p>{explanation}</p>")
        text_browser.setStyleSheet("background-color: #002a2a; border: none;")
        
        layout.addWidget(text_browser)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(detail_popup.close)
        close_button.setStyleSheet("""
            background-color: #001a1a;
            color: #00ff00;
            border: 2px solid #ff8c00;
            border-radius: 5px;
            padding: 5px;
        """)
        layout.addWidget(close_button)
        
        detail_popup.setLayout(layout)
        detail_popup.setGeometry(200, 200, 400, 300)
        detail_popup.exec()


    def start_flicker(self, component):
        if component not in self.flicker_timers:
            timer = QTimer(self)
            timer.timeout.connect(lambda: self.flicker_effect(component))
            timer.start(100)  # Flicker every 100ms
            self.flicker_timers[component] = timer


    def stop_flicker(self, component):
        if component in self.flicker_timers:
            self.flicker_timers[component].stop()
            del self.flicker_timers[component]
            component.setStyleSheet("border: 2px solid #ff8c00; background-color: #001a1a;")


    def flicker_effect(self, component):
        current_style = component.styleSheet()
        if "background-color: #001a1a" in current_style:
            component.setStyleSheet("border: 2px solid #ff8c00; background-color: #005a5a;")  # Brighter color
        else:
            component.setStyleSheet("border: 2px solid #ff8c00; background-color: #001a1a;")


    def update_display(self, data, purchase_price):
            print("Updating display with data:", data)
            try:
                # Safely get data with defaults
                self.current_stock_data = data.get('stock_data')
                current_price = data.get('current_price')
                company_name = data.get('company_name', 'N/A')
                self.nikkei_sentiment = data.get('nikkei_sentiment')
                self.yahoo_sentiment = data.get('yahoo_sentiment')
                self.nikkei_news_data = data.get('nikkei_news', [])
                self.yahoo_news_data = data.get('yahoo_news', [])
                psr = data.get('psr')
                pbr = data.get('pbr')
                roa = data.get('roa')
                roe = data.get('roe')
                next_day_prediction = data.get('next_day_prediction')

                # Get current price from stock data if it's not available directly
                if current_price is None and self.current_stock_data:
                    current_price = self.current_stock_data[0][1]  # Get the most recent price

                # Calculate overall sentiment only if both sentiments are numeric
                overall_sentiment = None
                if isinstance(self.nikkei_sentiment, (int, float)) and isinstance(self.yahoo_sentiment, (int, float)):
                    overall_sentiment = (self.nikkei_sentiment + self.yahoo_sentiment) / 2
                overall_sentiment_text = self.sentiment_to_text(overall_sentiment) if overall_sentiment is not None else "Insufficient data"

                # Pattern identification
                matched_pattern = self.identify_pattern(self.current_stock_data) if self.current_stock_data else "Unable to retrieve stock data"
                
                # Convert values to float where needed
                psr_val = float(psr) if psr is not None else None
                pbr_val = float(pbr) if pbr is not None else None
                roa_val = float(roa) if roa is not None else None
                roe_val = float(roe) if roe is not None else None

                # Get evaluations and recommendations
                psr_score, pbr_score, psr_comment, pbr_comment = self.evaluate_psr_pbr(psr_val, pbr_val)
                recommendation = self.get_action_recommendation(
                    overall_sentiment_text,
                    matched_pattern,
                    self.current_stock_data,
                    psr_val,
                    pbr_val,
                    roa_val,
                    roe_val,
                    purchase_price
                )

                # Update CASPER
                if next_day_prediction is not None and current_price is not None:
                    try:
                        prediction_change = ((float(next_day_prediction) - float(current_price)) / float(current_price)) * 100
                        next_day_prediction_text = f"¥{float(next_day_prediction):.2f} ({prediction_change:+.2f}%)"
                    except (ValueError, TypeError):
                        next_day_prediction_text = "Prediction not available"
                else:
                    next_day_prediction_text = "Prediction not available"

                casper_content = f"""
                    <p>Company: {company_name}</p>
                    <p><a href='nikkei'>Nikkei Sentiment: {self.sentiment_to_text(self.nikkei_sentiment)}</a></p>
                    <p><a href='yahoo'>Yahoo Sentiment: {self.sentiment_to_text(self.yahoo_sentiment)}</a></p>
                    <p>Overall Sentiment: {overall_sentiment_text}</p>
                    <p>Next Day Prediction: {next_day_prediction_text}</p>
                """
                casper_browser = self.casper.findChild(QTextBrowser)
                if self.has_content_changed('casper', casper_content):
                    self.update_component_with_flicker(self.casper, casper_content)
                else:
                    casper_browser.setHtml(casper_content)

                # Format metrics with proper conditional handling
                psr_display = f"{psr_val:.2f}" if psr_val is not None else "N/A"
                pbr_display = f"{pbr_val:.2f}" if pbr_val is not None else "N/A"
                roa_display = f"{roa_val:.2f}%" if roa_val is not None else "N/A"
                roe_display = f"{roe_val:.2f}%" if roe_val is not None else "N/A"

                # Update BALTHASAR
                recommendation_parts = recommendation.split('\n', 1)
                action = recommendation_parts[0]
                explanation = recommendation_parts[1] if len(recommendation_parts) > 1 else ""
                
                model_accuracy = data.get('model_accuracy')
                if isinstance(model_accuracy, (float, int)):
                    if 0 <= model_accuracy <= 1:
                        # Assuming accuracy is 1 - MAPE, format as percentage
                        model_accuracy_text = f"{model_accuracy:.2%}"
                    else:
                        # Could be an error or another metric
                        model_accuracy_text = f"Value: {model_accuracy:.2f}" 
                else:
                    model_accuracy_text = "Calculating..."

                key_metrics = f"""
                <p>PSR: {psr_display} | PBR: {pbr_display}</p>
                <p>ROA: {roa_display} | ROE: {roe_display}</p>
                <p>Model Accuracy: {model_accuracy_text}</p>
                """

                balthasar_content = f"""
                <p><strong>Recommendation: {action}</strong></p>
                <p>Pattern: {matched_pattern}</p>
                {key_metrics}
                <p><a href='detailed_explanation'>Click for detailed explanation</a></p>
                """
                
                balthasar_browser = self.balthasar.findChild(QTextBrowser)
                if self.has_content_changed('balthasar', balthasar_content):
                    self.update_component_with_flicker(self.balthasar, balthasar_content)
                else:
                    balthasar_browser.setHtml(balthasar_content)
                
                try:
                    balthasar_browser.anchorClicked.disconnect()
                except:
                    pass
                balthasar_browser.anchorClicked.connect(
                    lambda url: self.show_detailed_explanation(explanation) 
                    if url.toString() == 'detailed_explanation' else None
                )

                # Update MELCHIOR
                # Handle current price with proper type checking and conversion
                try:
                    current_price_text = f"¥{float(current_price):.2f}" if current_price not in (None, "", "N/A") else "N/A"
                except (ValueError, TypeError):
                    current_price_text = "N/A"
                    
                try:
                    purchase_price_text = f"¥{float(purchase_price):.2f}" if purchase_price not in (None, "", "N/A") else "N/A"
                except (ValueError, TypeError):
                    purchase_price_text = "N/A"
                
                melchior_content = f"""
                    <p>Current Price: {current_price_text}</p>
                    <p>Purchase Price: {purchase_price_text}</p>
                    <p>Price Difference: {self.calculate_price_difference(current_price, purchase_price)}</p>
                    <p>PSR: {psr_display} - {psr_comment}</p>
                    <p>PBR: {pbr_display} - {pbr_comment}</p>
                    <p>ROA: {roa_display}</p>
                    <p>ROE: {roe_display}</p>
                """
                melchior_browser = self.melchior.findChild(QTextBrowser)
                if self.has_content_changed('melchior', melchior_content):
                    self.update_component_with_flicker(self.melchior, melchior_content)
                else:
                    melchior_browser.setHtml(melchior_content)

                # Trigger the graph update
                self.graph_update_timer.start()

            except Exception as e:
                print(f"Error in update_display: {e}")
                self.show_error(f"An error occurred while updating the display: {e}")


    def handle_link_click(self, event, textedit):
        cursor = textedit.cursorForPosition(event.pos())
        if cursor.charFormat().isAnchor():
            anchor = cursor.charFormat().anchorHref()
            self.open_sentiment_popup(anchor)
        else:
            super(QTextEdit, textedit).mousePressEvent(event)


    def update_component_with_flicker(self, component, new_text):
        self.start_flicker(component)
        component.findChild(QTextBrowser).setHtml(new_text)
        QTimer.singleShot(500, lambda: self.stop_flicker(component))


    def show_error(self, message):
        for component in [self.casper, self.balthasar, self.melchior]:
            component.findChild(QTextEdit).setText(message)


    def sentiment_to_text(self, score):
        if score is None:
            return "No data"
        if score > 0.8:
            return "Very Negative"
        elif score > 0.6:
            return "Negative"
        elif score > 0.4:
            return "Neutral"
        elif score > 0.2:
            return "Positive"
        else:
            return "Very Positive"


    def identify_pattern(self, stock_data):
        if stock_data is None:
            return "No stock data available"
        
        if len(stock_data) < 5:  # Lowered from 30 to 5
            return f"Insufficient data for pattern identification (only {len(stock_data)} data points)"
        
        # Convert stock_data to DataFrame
        df = pd.DataFrame(stock_data, columns=['Date', 'Close'])
        df['Close'] = pd.to_numeric(df['Close'])
        
        # Calculate indicators
        df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = trend.MACD(df['Close']).macd_signal()
        df['BB_High'] = volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Low'] = volatility.BollingerBands(df['Close']).bollinger_lband()
        
        # Identify patterns
        patterns = []
        
        # Trend identification
        if len(df) >= 5:
            if df['Close'].iloc[-1] > df['Close'].iloc[-5]:
                patterns.append("Short-term Upward Trend")
            elif df['Close'].iloc[-1] < df['Close'].iloc[-5]:
                patterns.append("Short-term Downward Trend")
        
        # RSI overbought/oversold
        if not df['RSI'].isnull().all():
            if df['RSI'].iloc[-1] > 70:
                patterns.append("Overbought (RSI)")
            elif df['RSI'].iloc[-1] < 30:
                patterns.append("Oversold (RSI)")
        
        # MACD crossover
        if not df['MACD'].isnull().all() and not df['MACD_Signal'].isnull().all():
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
                patterns.append("Bullish MACD Crossover")
            elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
                patterns.append("Bearish MACD Crossover")
        
        # Bollinger Bands
        if not df['BB_High'].isnull().all() and not df['BB_Low'].isnull().all():
            if df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
                patterns.append("Price Above Upper Bollinger Band")
            elif df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
                patterns.append("Price Below Lower Bollinger Band")
        
        if patterns:
            return ", ".join(patterns)
        else:
            return "No clear pattern identified"


    def evaluate_psr_pbr(self, psr, pbr):
        psr_score = 0
        pbr_score = 0
        psr_comment = ""
        pbr_comment = ""
        
        if psr is not None:
            if psr > 4:
                psr_score = -1
                psr_comment = "Company may be overvalued based on PSR."
            elif 1 <= psr <= 2:
                psr_score = 1
                psr_comment = "Company may be undervalued based on PSR."
            else:
                psr_comment = "PSR is in a moderate range."
        else:
            psr_comment = "PSR data not available."
        
        if pbr is not None:
            if pbr > 3:
                pbr_score = -1
                pbr_comment = "Company may be overvalued based on PBR."
            elif pbr < 1:
                pbr_score = 1
                pbr_comment = "Company may be undervalued based on PBR."
            else:
                pbr_comment = "PBR is in a moderate range."
        else:
            pbr_comment = "PBR data not available."
        
        return psr_score, pbr_score, psr_comment, pbr_comment


    def stop_graph_updates(self):
        self.graph_update_timer.stop()


    def get_action_recommendation(self, public_opinion, stock_trend, stock_price_data, psr, pbr, roa, roe, purchase_price=None):
        if not stock_price_data:
            return "Insufficient data for recommendation"
        
        opinion_score = {"Very Positive": 2, "Positive": 1, "Neutral": 0, "Negative": -1, "Very Negative": -2}
        trend_score = {"Short-term Upward Trend": 1, "Short-term Downward Trend": -1, "No clear pattern identified": 0}
        
        psr_score, pbr_score, _, _ = self.evaluate_psr_pbr(psr, pbr)
        
        # Evaluate ROA and ROE
        roa_score = 1 if roa and roa > 5 else (-1 if roa and roa < 2 else 0)
        roe_score = 1 if roe and roe > 15 else (-1 if roe and roe < 8 else 0)
        
        # Base total score
        total_score = (
            opinion_score.get(public_opinion, 0) + 
            trend_score.get(stock_trend, 0) + 
            psr_score + 
            pbr_score +
            roa_score +
            roe_score
        )
        
        prices = [price for _, price in stock_price_data]
        current_price = prices[0]
        avg_price = np.mean(prices)
        std_dev = np.std(prices)
        
        # User owns the stock if purchase_price is not None
        owns_stock = purchase_price is not None
        
        if owns_stock:
            price_change = (current_price - purchase_price) / purchase_price * 100
            
            if total_score > 2:
                action = "Hold"
                explanation = (
                    f"Strong positive outlook. You're currently {'up' if price_change > 0 else 'down'} "
                    f"{abs(price_change):.2f}%. Consider holding for potential further gains."
                )
            elif total_score < -2:
                action = "Consider Selling"
                explanation = (
                    f"Strong negative outlook. You're currently {'up' if price_change > 0 else 'down'} "
                    f"{abs(price_change):.2f}%. Consider selling to {'lock in profits' if price_change > 0 else 'minimize losses'}."
                )
            else:
                action = "Hold and Monitor"
                explanation = (
                    f"Mixed signals. You're currently {'up' if price_change > 0 else 'down'} "
                    f"{abs(price_change):.2f}%. Monitor the stock closely for changes in sentiment or market trends."
                )
        else:
            if total_score > 2:
                target_price = max(current_price * 0.99, avg_price - 0.5 * std_dev)
                action = f"Consider Buying (Target: ¥{target_price:.2f})"
                explanation = "Strong positive outlook. Consider buying near the suggested target price."
            elif total_score < -2:
                action = "Hold Off"
                explanation = "Strong negative outlook. It might be better to wait for a more favorable entry point."
            else:
                action = "Monitor"
                explanation = "Mixed signals. Monitor the stock for a clearer trend before making a decision."

        # Add technical analysis explanation
        explanation += f"\n\nThis recommendation is based on:"
        explanation += f"\nSentiment: {public_opinion}"
        explanation += f"\nTrend: {stock_trend}"
        explanation += f"\nPSR: {psr:.2f}"
        explanation += f"\nPBR: {pbr:.2f}"
        explanation += f"\nROA: {roa:.2f}%"
        explanation += f"\nROE: {roe:.2f}%"
            
        return f"{action}\nExplanation: {explanation}"


    def calculate_price_difference(self, current_price, purchase_price):
        if purchase_price is None:
            return "N/A"
        price_difference = current_price - purchase_price
        price_percentage = (price_difference / purchase_price) * 100
        return f"¥{price_difference:.2f} ({price_percentage:.2f}%)"


    def closeEvent(self, event):
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread.wait()
        event.accept()



class ModelTracker:
    def __init__(self):
        self.accuracy_history = []
        self.prediction_history = []
        self.best_model = None
        self.best_accuracy = 0
        self.model_path = 'best_model.pkl'
        self.selected_features = None  # Add this to store selected features

        # Load the model if it exists
        if os.path.exists(self.model_path):
            self.load_model()  # Automatically load on initialization

    def save_model(self, model, accuracy, scaler=None, selected_features=None):
        """Saves the model if it's better than the current best model."""
        if accuracy > self.best_accuracy:
            self.best_model = model
            self.best_accuracy = accuracy
            self.scaler = scaler
            self.selected_features = selected_features  # Store selected features

            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'accuracy': accuracy,
                        'scaler': scaler,
                        'selected_features': selected_features  # Save selected features
                    }, f)
                print(f"Saved new best model with accuracy: {accuracy}")

                # Verify saved content
                try:
                    with open(self.model_path, 'rb') as f:
                        saved_data = pickle.load(f)
                    if 'scaler' in saved_data:
                        print("Scaler successfully saved")
                    else:
                        print("Warning: Scaler not saved properly")
                    if 'selected_features' in saved_data:
                        print("Selected features successfully saved")
                    else:
                        print("Warning: Selected features not saved properly")
                except Exception as e:
                    print(f"Warning: Could not verify saved model: {e}")

            except Exception as e:
                print(f"Error saving model: {e}")

    def load_model(self):
        """Loads a saved model if it exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.best_model = saved_data['model']
                    self.best_accuracy = saved_data['accuracy']
                    self.scaler = saved_data.get('scaler')  # Load scaler
                    self.selected_features = saved_data.get('selected_features') # Load selected features
                    print(f"Loaded saved model with accuracy: {self.best_accuracy}")
                    if self.scaler:
                        print("Scaler loaded successfully")
                    if self.selected_features:
                        print("Selected features loaded successfully")
                    return True
            except Exception as e:
                print(f"Error loading saved model: {e}")
                traceback.print_exc()
        return False


    def track_performance(self, actual, predicted):
        """予測性能を追跡"""
        try:
            accuracy = 1 - mean_absolute_percentage_error([actual], [predicted])
            self.accuracy_history.append(accuracy)
            self.prediction_history.append({
                'actual': actual,
                'predicted': predicted,
                'timestamp': datetime.now()
            })
        except Exception as e:
            print(f"Error tracking performance: {e}")

    def get_average_accuracy(self, window=30):
        """直近の精度の平均を計算"""
        if len(self.accuracy_history) == 0:
            return None
        recent = self.accuracy_history[-window:]
        return sum(recent) / len(recent)

    def get_prediction_statistics(self):
        """予測の統計情報を取得"""
        if len(self.prediction_history) == 0:
            return None
        
        actuals = [p['actual'] for p in self.prediction_history]
        predicteds = [p['predicted'] for p in self.prediction_history]
        
        return {
            'mape': mean_absolute_percentage_error(actuals, predicteds),
            'rmse': np.sqrt(mean_squared_error(actuals, predicteds)),
            'predictions_count': len(self.prediction_history)
        }



if __name__ == '__main__':
    print("Starting MAGI Stock Analysis application...")
    app = QApplication(sys.argv)
    ex = MAGIStockAnalysis()
    ex.show()
    sys.exit(app.exec())
