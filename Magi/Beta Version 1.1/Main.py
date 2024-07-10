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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import atexit
import tempfile




def cleanup():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('_MEI') or filename.startswith('magi_'):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except:
                pass

atexit.register(cleanup)



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
        self.ml_model = None


    def get_jpx_nikkei_data(self):
        # JPX-Nikkei Index 400 ticker
        jpx_nikkei_ticker = "^NKJ400"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        
        try:
            df = yf.download(jpx_nikkei_ticker, start=start_date, end=end_date)
            
            # Calculate additional features
            df['JPX_Returns'] = df['Close'].pct_change()
            df['JPX_Volatility'] = df['Close'].rolling(window=30).std()
            
            return df
        except Exception as e:
            print(f"Error retrieving JPX-Nikkei Index 400 data: {e}")
            return None


    def train_ml_model(self):
        # Fetch historical data for the stock
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        df = yf.download(f"{self.stock_number}.T", start=start_date, end=end_date)
        # Feature engineering
        df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = trend.MACD(df['Close']).macd()
        df['BB_High'] = volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Low'] = volatility.BollingerBands(df['Close']).bollinger_lband()
        df['Return'] = df['Close'].pct_change()
        
        # Prepare features and target
        features = ['Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Return']
        
        # Shift the target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        X = df[features]
        y = df['Target']
        
        # Ensure X and y have the same number of samples
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler


    def predict_next_day_price(self, current_data):
        # Prepare the input data
        input_data = np.array([
            current_data['Open'], 
            current_data['High'], 
            current_data['Low'], 
            current_data['Close'],
            current_data['RSI'], 
            current_data['MACD'], 
            current_data['BB_High'], 
            current_data['BB_Low'], 
            current_data['Return']
        ])
        input_data = input_data.reshape(1, -1)

        # Make prediction
        prediction = self.ml_model.predict(input_data)
        return prediction[0]


    def run(self):
        while self.running:
            data = self.fetch_latest_data()
            
            # Train the model for each run
            self.ml_model, _ = self.train_ml_model()
            
            # Add ML prediction
            current_price = data['current_price']
            stock_data = data['stock_data']
            if stock_data:
                dates, prices = zip(*stock_data)
                latest_data = {
                    'Open': prices[0],
                    'High': max(prices),
                    'Low': min(prices),
                    'Close': current_price,
                    'RSI': momentum.RSIIndicator(pd.Series(prices)).rsi().iloc[-1],
                    'MACD': trend.MACD(pd.Series(prices)).macd().iloc[-1],
                    'BB_High': volatility.BollingerBands(pd.Series(prices)).bollinger_hband().iloc[-1],
                    'BB_Low': volatility.BollingerBands(pd.Series(prices)).bollinger_lband().iloc[-1],
                    'Return': (current_price - prices[1]) / prices[1] if len(prices) > 1 else 0
                }
                next_day_prediction = self.predict_next_day_price(latest_data)
                data['next_day_prediction'] = next_day_prediction
            
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



class MAGIStockAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.nikkei_news_data = []
        self.yahoo_news_data = []
        
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.ja_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.ja_model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")
        self.en_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.en_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.flicker_timers = {}
        self.update_thread = None
        self.previous_results = {
            'casper': '',
            'balthasar': '',
            'melchior': ''
        }

        self.showFullScreen()
        self.setWindowState(Qt.WindowState.WindowFullScreen)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.showNormal()
        super().keyPressEvent(event)


    def analyze_stock(self):
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

        # Start flickering
        for component in [self.casper, self.balthasar, self.melchior]:
            self.start_flicker(component)
            component.findChild(QTextBrowser).setText("Loading...")

        # Stop existing update thread if running
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread.wait()

        # Start new update thread
        self.update_thread = StockUpdateThread(stock_number, self.ja_tokenizer, self.ja_model, self.en_tokenizer, self.en_model)
        self.update_thread.update_signal.connect(lambda data: self.update_display(data, purchase_price))
        self.update_thread.start()


    def has_content_changed(self, component_name, new_content):
        if self.previous_results[component_name] != new_content:
            self.previous_results[component_name] = new_content
            return True
        return False


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
        try:
            current_price = data['current_price']
            company_name = data['company_name']
            self.nikkei_sentiment = data['nikkei_sentiment']
            self.yahoo_sentiment = data['yahoo_sentiment']
            self.nikkei_news_data = data['nikkei_news']
            self.yahoo_news_data = data['yahoo_news']
            stock_data = data['stock_data']
            psr, pbr = data['psr'], data['pbr']
            roa = data['roa']
            roe = data['roe']
            next_day_prediction = data.get('next_day_prediction')  # Get the prediction

            overall_sentiment = (self.nikkei_sentiment + self.yahoo_sentiment) / 2 if self.nikkei_sentiment is not None and self.yahoo_sentiment is not None else None
            overall_sentiment_text = self.sentiment_to_text(overall_sentiment) if overall_sentiment is not None else "Insufficient data"

            matched_pattern = self.identify_pattern(stock_data) if stock_data else "Unable to retrieve stock data"
            psr_score, pbr_score, psr_comment, pbr_comment = self.evaluate_psr_pbr(psr, pbr)
            recommendation = self.get_action_recommendation(overall_sentiment_text, matched_pattern, stock_data, psr, pbr, roa, roe, purchase_price)

            # Update CASPER
            next_day_prediction_text = f"¥{next_day_prediction:.2f}" if next_day_prediction is not None else "N/A"
            prediction_change = ((next_day_prediction - current_price) / current_price * 100) if next_day_prediction is not None and current_price is not None else None
            prediction_change_text = f"{prediction_change:.2f}%" if prediction_change is not None else "N/A"

            casper_content = f"""
                <p>Company: {company_name}</p>
                <p><a href='nikkei'>Nikkei Sentiment: {self.sentiment_to_text(self.nikkei_sentiment)}</a></p>
                <p><a href='yahoo'>Yahoo Sentiment: {self.sentiment_to_text(self.yahoo_sentiment)}</a></p>
                <p>Overall Sentiment: {overall_sentiment_text}</p>
                <p>Next Day Prediction: {next_day_prediction_text} ({prediction_change_text})</p>
            """
            casper_browser = self.casper.findChild(QTextBrowser)
            if self.has_content_changed('casper', casper_content):
                self.update_component_with_flicker(self.casper, casper_content)
            else:
                casper_browser.setHtml(casper_content)

            # Update BALTHASAR
            balthasar_browser = self.balthasar.findChild(QTextBrowser)
            
            recommendation_parts = recommendation.split('\n', 1)
            action = recommendation_parts[0]
            explanation = recommendation_parts[1] if len(recommendation_parts) > 1 else ""
            
            key_metrics = f"""
            <p>PSR: {psr:.2f} | PBR: {pbr:.2f}</p>
            <p>ROA: {roa:.2f}% | ROE: {roe:.2f}%</p>
            """

            balthasar_content = f"""
            <p><strong>Recommendation: {action}</strong></p>
            <p>Pattern: {matched_pattern}</p>
            {key_metrics}
            <p><a href='detailed_explanation'>Click for detailed explanation</a></p>
            """
            
            if self.has_content_changed('balthasar', balthasar_content):
                self.update_component_with_flicker(self.balthasar, balthasar_content)
                self.update_graph(self.balthasar, stock_data)
            else:
                balthasar_browser.setHtml(balthasar_content)
            
            # Disconnect any existing connections to avoid multiple connections
            try:
                balthasar_browser.anchorClicked.disconnect()
            except:
                pass
            balthasar_browser.anchorClicked.connect(lambda url: self.show_detailed_explanation(explanation) if url.toString() == 'detailed_explanation' else None)

            # Update MELCHIOR
            current_price_text = f"¥{current_price:.2f}" if current_price is not None else "N/A"
            purchase_price_text = f"¥{purchase_price:.2f}" if purchase_price is not None else "N/A"
            psr_text = f"{psr:.2f}" if psr is not None else "N/A"
            pbr_text = f"{pbr:.2f}" if pbr is not None else "N/A"
            roa_text = f"{roa:.2f}%" if roa is not None else "N/A"
            roe_text = f"{roe:.2f}%" if roe is not None else "N/A"
            
            melchior_content = f"""
                <p>Current Price: {current_price_text}</p>
                <p>Purchase Price: {purchase_price_text}</p>
                <p>Price Difference: {self.calculate_price_difference(current_price, purchase_price)}</p>
                <p>PSR: {psr_text} - {psr_comment}</p>
                <p>PBR: {pbr_text} - {pbr_comment}</p>
                <p>ROA: {roa_text}</p>
                <p>ROE: {roe_text}</p>
            """
            melchior_browser = self.melchior.findChild(QTextBrowser)
            if self.has_content_changed('melchior', melchior_content):
                self.update_component_with_flicker(self.melchior, melchior_content)
            else:
                melchior_browser.setHtml(melchior_content)

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


    def update_graph(self, component, stock_data):
        if stock_data:
            dates = [date for date, _ in stock_data]
            prices = [price for _, price in stock_data]

            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#001a1a')
            ax.plot(dates, prices, color='#00ff00')
            ax.set_facecolor('#001a1a')
            ax.tick_params(axis='x', colors='#00ff00')
            ax.tick_params(axis='y', colors='#00ff00')
            ax.spines['bottom'].set_color('#00ff00')
            ax.spines['top'].set_color('#00ff00')
            ax.spines['left'].set_color('#00ff00')
            ax.spines['right'].set_color('#00ff00')
            plt.title('30-Day Stock Pattern', color='#00ff00')
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



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MAGIStockAnalysis()
    ex.show()
    sys.exit(app.exec())
