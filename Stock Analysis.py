import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime, timedelta
import json
from langdetect import detect
import pandas as pd
import yfinance as yf

def get_company_name(stock_number):
    url = f"https://finance.yahoo.co.jp/quote/{stock_number}.T"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.text.strip()
        company_name = title.split('【')[0].strip()
        return company_name
    else:
        return None

def get_current_stock_price(stock_number):
    url = f"https://finance.yahoo.co.jp/quote/{stock_number}.T"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    price_element = soup.select_one('span._3rXWJKZF')
    
    if price_element:
        price_text = price_element.text.strip().replace(',', '')
        try:
            return float(price_text)
        except ValueError:
            return None
    else:
        return None

def scrape_nikkei_news(stock_number):
    url = f"https://www.nikkei.com/nkd/company/news/?scode={stock_number}&ba=1"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('a', href=lambda href: href and "/nkd/company/article/" in href)
    news_data = []
    for item in news_items:
        title = item.text.strip()
        url = "https://www.nikkei.com" + item['href']
        news_data.append({"title": title, "url": url})
    return news_data

def scrape_yahoo_finance_news(stock_number):
    ticker = f"{stock_number}.T"
    url = f"https://finance.yahoo.co.jp/quote/{ticker}/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('a', href=lambda href: href and "/news/" in href)
    news_data = []
    for item in news_items:
        title = item.text.strip()
        article_url = item['href']
        if not article_url.startswith('http'):
            article_url = "https://finance.yahoo.co.jp" + article_url
        news_data.append({"title": title, "url": article_url})
    return news_data

def analyze_sentiment(text, ja_tokenizer, ja_model, en_tokenizer, en_model):
    try:
        lang = detect(text)
    except:
        lang = 'ja'  # Default to Japanese if detection fails

    if lang == 'ja':
        tokenizer = ja_tokenizer
        model = ja_model
    else:
        tokenizer = en_tokenizer
        model = en_model

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    sentiment_score = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return sentiment_score[0]  # Return the raw sentiment score

def sentiment_to_text(score):
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

def get_stock_data(stock_number):
    ticker = f"{stock_number}.T"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print("No data found for the specified stock number.")
            return None
        
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data = [(row['Date'], row['Close']) for _, row in df.iterrows()]
        
        # Sort by date (newest first) and return up to 30 days of data
        stock_data.sort(key=lambda x: x[0], reverse=True)
        return stock_data[:30]
    
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None

def identify_pattern(stock_data):
    if stock_data is None or len(stock_data) < 5:
        return "Insufficient data for pattern identification"
    
    prices = [price for _, price in stock_data]
    dates = [date for date, _ in stock_data]
    
    # Reverse the lists to have oldest data first
    prices = prices[::-1]
    dates = dates[::-1]
    
    n = len(prices)
    changes = np.diff(prices)
    
    def is_increasing(data):
        return np.all(np.diff(data) >= 0)
    
    def is_decreasing(data):
        return np.all(np.diff(data) <= 0)
    
    def find_peaks(data, order=3):
        peaks = []
        for i in range(order, len(data) - order):
            if all(data[i] > data[i-j] for j in range(1, order+1)) and all(data[i] > data[i+j] for j in range(1, order+1)):
                peaks.append(i)
        return np.array(peaks)
    
    def find_troughs(data, order=3):
        troughs = []
        for i in range(order, len(data) - order):
            if all(data[i] < data[i-j] for j in range(1, order+1)) and all(data[i] < data[i+j] for j in range(1, order+1)):
                troughs.append(i)
        return np.array(troughs)
    
    # Upward and Downward Trends
    if is_increasing(prices):
        return "Upward Trend"
    elif is_decreasing(prices):
        return "Downward Trend"
    
    # V-Shape Recovery and Inverted V-Shape
    if n >= 5:
        if prices[0] > prices[1] > prices[2] < prices[3] < prices[4]:
            return "V-Shape Recovery"
        elif prices[0] < prices[1] < prices[2] > prices[3] > prices[4]:
            return "Inverted V-Shape"
    
    # Double Bottom and Double Top
    peaks = find_peaks(prices)
    troughs = find_troughs(prices)
    
    if len(troughs) >= 2 and troughs[-1] - troughs[-2] >= 5:
        if abs(prices[troughs[-1]] - prices[troughs[-2]]) / prices[troughs[-2]] < 0.03:
            return "Double Bottom"
    
    if len(peaks) >= 2 and peaks[-1] - peaks[-2] >= 5:
        if abs(prices[peaks[-1]] - prices[peaks[-2]]) / prices[peaks[-2]] < 0.03:
            return "Double Top"
    
    # Head and Shoulders
    if len(peaks) >= 3:
        if prices[peaks[1]] > prices[peaks[0]] and prices[peaks[1]] > prices[peaks[2]]:
            if abs(prices[peaks[0]] - prices[peaks[2]]) / prices[peaks[0]] < 0.03:
                return "Head and Shoulders"
    
    # Triangles and Wedges
    if n >= 15:
        first_half = prices[:n//2]
        second_half = prices[n//2:]
        
        if is_increasing(first_half) and is_decreasing(second_half):
            return "Ascending Triangle"
        elif is_decreasing(first_half) and is_increasing(second_half):
            return "Descending Triangle"
        elif (max(first_half) > max(second_half) and min(first_half) < min(second_half)):
            return "Symmetrical Triangle"
        elif (max(first_half) > max(second_half) and min(first_half) > min(second_half)):
            return "Falling Wedge"
        elif (max(first_half) < max(second_half) and min(first_half) < min(second_half)):
            return "Rising Wedge"
    
    # Pennant and Flag
    if n >= 20:
        if is_increasing(prices[:5]) and np.all(np.abs(np.diff(prices[5:])) < np.mean(np.abs(np.diff(prices[:5])))):
            return "Bullish Pennant"
        elif is_decreasing(prices[:5]) and np.all(np.abs(np.diff(prices[5:])) < np.mean(np.abs(np.diff(prices[:5])))):
            return "Bearish Pennant"
        elif is_increasing(prices[:10]) and is_decreasing(prices[10:]):
            return "Bullish Flag"
        elif is_decreasing(prices[:10]) and is_increasing(prices[10:]):
            return "Bearish Flag"
    
    # Rounding Bottom and Top
    if n >= 15:
        first_third = prices[:n//3]
        last_third = prices[-n//3:]
        if is_decreasing(first_third) and is_increasing(last_third):
            return "Rounding Bottom"
        elif is_increasing(first_third) and is_decreasing(last_third):
            return "Rounding Top"
    
    return "No specific pattern identified"

def get_suggested_price(stock_number):
    ticker = f"{stock_number}.T"
    stock = yf.Ticker(ticker)
    history = stock.history(period="1y")
    
    if history.empty:
        return None, None
    
    current_price = history['Close'][-1]
    avg_price = history['Close'].mean()
    max_price = history['Close'].max()
    min_price = history['Close'].min()
    
    buy_price = (avg_price + min_price) / 2
    sell_price = (avg_price + max_price) / 2
    
    return round(buy_price, 2), round(sell_price, 2)

# Main function to get all the information
def get_stock_info(stock_number):
    company_name = get_company_name(stock_number)
    current_stock_price = get_current_stock_price(stock_number)
    
    ja_tokenizer = AutoTokenizer.from_pretrained("daigo/bert-base-japanese-sentiment")
    ja_model = AutoModelForSequenceClassification.from_pretrained("daigo/bert-base-japanese-sentiment")
    en_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    en_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    
    news_data = scrape_nikkei_news(stock_number) + scrape_yahoo_finance_news(stock_number)
    
    news_sentiments = []
    for news in news_data:
        sentiment_score = analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model)
        sentiment_text = sentiment_to_text(sentiment_score)
        news_sentiments.append({
            'title': news['title'],
            'url': news['url'],
            'sentiment': sentiment_text
        })
    
    stock_data = get_stock_data(stock_number)
    pattern = identify_pattern(stock_data)
    
    buy_price, sell_price = get_suggested_price(stock_number)
    
    return {
        'company_name': company_name,
        'current_stock_price': current_stock_price,
        'news': news_sentiments,
        'pattern': pattern,
        'buy_price': buy_price,
        'sell_price': sell_price
    }

def get_action_recommendation(public_opinion, stock_trend, stock_price_data, purchase_price=None):
    if not stock_price_data:
        return "Insufficient data for recommendation"

    opinion_score = {"Very Positive": 2, "Positive": 1, "Neutral": 0, "Negative": -1, "Very Negative": -2}
    trend_score = {"V-Shape Recovery": 1, "Upward Trend": 1, "Downward Trend": -1, "No specific pattern identified": 0}
    
    total_score = opinion_score.get(public_opinion, 0) + trend_score.get(stock_trend, 0)
    
    prices = [price for _, price in stock_price_data]
    current_price = prices[0]
    avg_price = np.mean(prices)
    std_dev = np.std(prices)
    
    if total_score > 0:
        target_price = max(current_price * 0.99, avg_price - 0.5 * std_dev)
        action = f"Buy (Target: ¥{target_price:.2f})"
        explanation = "Positive outlook. Consider buying near the suggested target price."
    elif total_score < 0:
        target_price = min(current_price * 1.01, avg_price + 0.5 * std_dev)
        action = f"Sell (Target: ¥{target_price:.2f})"
        explanation = "Negative outlook. Consider selling near the suggested target price."
    else:
        action = "Hold"
        explanation = "Mixed signals. Monitor the stock closely for changes in sentiment or market trends."

    if purchase_price:
        price_change = (current_price - purchase_price) / purchase_price * 100
        if price_change < -5 and action.startswith("Sell"):
            explanation += f" Caution: Selling would result in a {abs(price_change):.2f}% loss from your purchase price."
        elif price_change > 5 and action == "Hold":
            explanation += f" Consider taking profits, current gain: {price_change:.2f}%."

    return f"{action}\nExplanation: {explanation}"

def main():
    while True:
        stock_input = input("Enter the stock exchange number (e.g., 3092 for ZOZO): ").strip().upper()
        
        if not stock_input.isdigit():
            print("Please enter a valid numeric stock code.")
            continue

        stock_number = stock_input
        company_name = get_company_name(stock_number)
        
        if company_name:
            confirmation = input(f"The company for stock number {stock_number} is {company_name}. Is this correct? (y/n): ")
            if confirmation.lower() == 'y':
                break
            else:
                print("Let's try again with a different stock number.")
        else:
            print("Company name could not be found. Please enter a valid stock exchange number.")

    purchase_price = input("Enter the price of the stock when you bought it (if not purchased, enter 'n/a'): ")
    if purchase_price.lower() == 'n/a':
        purchase_price = None
    else:
        try:
            purchase_price = float(purchase_price)
        except ValueError:
            print("Invalid price entered. Setting purchase price to N/A.")
            purchase_price = None

    ja_tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")
    ja_model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")
    
    en_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    en_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    try:
        nikkei_news_data = scrape_nikkei_news(stock_number)
        yahoo_finance_news_data = scrape_yahoo_finance_news(stock_number)
        
        nikkei_sentiments = [analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model) for news in nikkei_news_data]
        yahoo_finance_sentiments = [analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model) for news in yahoo_finance_news_data]
        
        def get_overall_sentiment(sentiments):
            if not sentiments:
                return "No data"
            avg_sentiment = sum(sentiments) / len(sentiments)
            return sentiment_to_text(avg_sentiment)
        
        nikkei_overall_sentiment = get_overall_sentiment(nikkei_sentiments)
        yahoo_finance_overall_sentiment = get_overall_sentiment(yahoo_finance_sentiments)
        
        if nikkei_sentiments and yahoo_finance_sentiments:
            overall_sentiment_value = (sum(nikkei_sentiments) / len(nikkei_sentiments) + sum(yahoo_finance_sentiments) / len(yahoo_finance_sentiments)) / 2
            overall_sentiment = sentiment_to_text(overall_sentiment_value)
        else:
            overall_sentiment = "Insufficient data"
        
        stock_data = get_stock_data(stock_number)
        if stock_data:
            matched_pattern = identify_pattern(stock_data)
        else:
            matched_pattern = "Unable to retrieve stock data"
        
        current_stock_price = get_current_stock_price(stock_number)
        
        if current_stock_price is not None:
            print(f"\nStock Analysis for {company_name} ({stock_number}):")
            print(f"Current Price: ¥{current_stock_price:.2f}")
            
            if purchase_price is not None:
                price_difference = current_stock_price - purchase_price
                price_percentage = (price_difference / purchase_price) * 100
                print(f"Purchase Price: ¥{purchase_price:.2f}")
                print(f"Price Difference: ¥{price_difference:.2f} ({price_percentage:.2f}%)")
            else:
                print("Purchase Price: N/A")
                print("Price Difference: N/A")
            
            print(f"\nIdentified 30-Day Pattern: {matched_pattern}")
            print(f"\nNikkei Overall Sentiment: {nikkei_overall_sentiment}")
            print(f"Yahoo Finance Overall Sentiment: {yahoo_finance_overall_sentiment}")
            print(f"Overall Sentiment: {overall_sentiment}")
            
            recommendation = get_action_recommendation(overall_sentiment, matched_pattern, stock_data, purchase_price)
            print(f"\nRecommended Action: {recommendation}")
            
            while True:
                source_request = input("\nEnter a news source name to see detailed sentiment analysis (or 'quit' to exit): ")
                if source_request.lower() == 'quit':
                    break
                
                news_sources = {
                    "Nikkei": list(zip(nikkei_news_data, map(sentiment_to_text, nikkei_sentiments))),
                    "Yahoo Finance": list(zip(yahoo_finance_news_data, map(sentiment_to_text, yahoo_finance_sentiments)))
                }
                
                if source_request in news_sources:
                    print(f"\nDetailed Sentiment Analysis for {source_request}:")
                    for article, sentiment in news_sources[source_request]:
                        print(f"Title: {article['title']}")
                        print(f"Sentiment: {sentiment}")
                        print(f"URL: {article['url']}")
                        print()
                else:
                    print("Invalid source name. Available sources: Nikkei, Yahoo Finance")
        else:
            print("Unable to retrieve current stock price. Please check the stock number and try again.")
    except Exception as e:
        print(f"An error occurred while processing the stock data: {str(e)}")
        print("Please check the stock number and try again.")

if __name__ == '__main__':
    main()
