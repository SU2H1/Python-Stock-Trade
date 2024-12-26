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


def cleanup():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('_MEI') or filename.startswith('magi_'):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except:
                pass
atexit.register(cleanup)



class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("LSTM training timed out")



class StockUpdateThread(QThread):
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

    def train_ml_model(self):
        # Fetch historical data for the stock
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)  # 10 years of data
        df = yf.download(f"{self.stock_number}.T", start=start_date, end=end_date)

        # Fetch Nikkei 225 data
        nikkei_df = yf.download("^N225", start=start_date, end=end_date)
        df['Nikkei_Return'] = nikkei_df['Close'].pct_change()

        # Feature engineering
        df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = trend.MACD(df['Close']).macd()
        df['BB_High'] = volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Low'] = volatility.BollingerBands(df['Close']).bollinger_lband()
        df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        df['Return'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Improved data preprocessing
        df['Volume'] = np.log1p(df['Volume'])  # Log-transform volume
        df['Volume_Change'] = df['Volume_Change'].clip(-1, 5)  # Cap between -100% and 500%
        df = df[df['Volume'] > 0]  # Remove days with zero volume
        
        # Prepare features and target
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'VWAP', 'Return', 'Volume_Change', 'MA_5', 'MA_20', 'Nikkei_Return']
        
        # Shift the target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        X = df[features]
        y = df['Target']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with the median of the respective column
        X = X.fillna(X.median())
        
        # Ensure X and y have the same number of samples
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        
        # Train the improved predictor
        self.predictor = ImprovedStockPredictor()
        self.predictor.train(X, y)
        
        # Evaluate the model on the entire dataset
        X_processed, _ = self.predictor.preprocess_data(X)
        y_pred = self.predictor.predict(X_processed)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        print(f"Model RMSE on entire dataset: {rmse}")
        
        return self.predictor

    def predict_next_day_price(self, current_data):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'VWAP', 'Return', 'Volume_Change', 'MA_5', 'MA_20', 'Nikkei_Return']
        input_data = current_data[features].values.reshape(1, -1)
        
        # Handle potential infinite values
        input_data = np.where(np.isinf(input_data), np.nan, input_data)
        input_data = np.nan_to_num(input_data, nan=np.nanmean(input_data))  # Replace NaN with mean
        
        # Make prediction using the improved predictor
        prediction = self.predictor.predict(input_data)[0]
        
        # Sanity check: limit the prediction to a reasonable range (e.g., ±5% of current price)
        current_price = current_data['Close'].values[0]
        max_change = 0.05  # 5% maximum change
        final_prediction = np.clip(prediction, current_price * (1 - max_change), current_price * (1 + max_change))
        
        return final_prediction

    def run(self):
        while self.running:
            print("Fetching latest data...")
            data = self.fetch_latest_data()
            print(f"Data fetched: {data}")
            
            if self.predictor is None:
                print("Training model...")
                try:
                    self.predictor = self.train_ml_model()
                    print("Model training completed")
                except Exception as e:
                    print(f"Error in model training: {e}")
                    self.predictor = None  # Reset predictor if training fails
            
            if self.predictor:
                current_price = data['current_price']
                stock_data = data['stock_data']
                if stock_data:
                    print("Preparing prediction data...")
                    dates, prices = zip(*stock_data)
                    latest_data = self.prepare_prediction_data(dates, prices, current_price)
                    print("Making prediction...")
                    next_day_prediction = self.predict_next_day_price(latest_data)
                    data['next_day_prediction'] = next_day_prediction
                    print(f"Next day prediction: {next_day_prediction}")
            
            print("Emitting update signal...")
            self.update_signal.emit(data)
            
            time.sleep(1)  # Update every second


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
            'roe': roe
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

    def prepare_prediction_data(self, dates, prices, current_price):
        print("Preparing prediction data...")
        df = pd.DataFrame({'Date': dates, 'Close': prices})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close'].rolling(window=2).max()
        df['Low'] = df['Close'].rolling(window=2).min()
        df['Volume'] = np.random.randint(1000000, 10000000, size=len(df))  # Dummy volume data
        
        # Calculate features
        df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = trend.MACD(df['Close']).macd()
        df['BB_High'] = volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Low'] = volatility.BollingerBands(df['Close']).bollinger_lband()
        df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        df['Return'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Nikkei_Return'] = 0  # Placeholder, ideally you'd fetch real Nikkei data

        # Use the most recent data point
        latest_data = df.iloc[-1:].copy()  # Create a copy to avoid SettingWithCopyWarning
        latest_data.loc[latest_data.index[-1], 'Close'] = current_price  # Use the current price
        print("Prediction data prepared:", latest_data)
        return latest_data

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
            
            model_accuracy = data.get('model_accuracy', 'N/A')
            model_accuracy_text = f"{model_accuracy:.2%}" if isinstance(model_accuracy, float) else model_accuracy
            
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
                explanation = f"Strong positive outlook. You're currently {'up' if price_change > 0 else 'down'} {abs(price_change):.2f}%. Consider holding for potential further gains."
            elif total_score < -2:
                action = "Consider Selling"
                explanation = f"Strong negative outlook. You're currently {'up' if price_change > 0 else 'down'} {abs(price_change):.2f}%. Consider selling to {'lock in profits' if price_change > 0 else 'minimize losses'}."
            else:
                action = "Hold and Monitor"
                explanation = f"Mixed signals. You're currently {'up' if price_change > 0 else 'down'} {abs(price_change):.2f}%. Monitor the stock closely for changes in sentiment or market trends."
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

        explanation += f"\n\nThis recommendation is based on:\nSentiment: {public_opinion}\nTrend: {stock_trend}\nPSR: {psr:.2f}\nPBR: {pbr:.2f}\nROA: {roa:.2f}%\nROE: {roe:.2f}%"
        
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
    def __init__(self, window_size=30):
        self.predictions = []
        self.actuals = []
        self.window_size = window_size

    def add_prediction(self, prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

    def add_actual(self, actual):
        self.actuals.append(actual)
        if len(self.actuals) > self.window_size:
            self.actuals.pop(0)

    def calculate_accuracies(self):
        return [1 - abs(p - a) / a for p, a in zip(self.predictions, self.actuals)]

    def calculate_accuracy(self):
        if len(self.predictions) != len(self.actuals) or len(self.predictions) < 2:
            return None
        
        mape = mean_absolute_percentage_error(self.actuals, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'Accuracy': 1 - mape  # This is a rough approximation of accuracy
        }



if __name__ == '__main__':
    print("Starting MAGI Stock Analysis application...")
    app = QApplication(sys.argv)
    ex = MAGIStockAnalysis()
    ex.show()
    sys.exit(app.exec())
