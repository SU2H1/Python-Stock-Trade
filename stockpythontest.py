import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

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
        url = "https://finance.yahoo.co.jp" + item['href']
        news_data.append({"title": title, "url": url})
    return news_data

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    sentiment_score = torch.softmax(outputs.logits, dim=1).tolist()[0]
    if sentiment_score[0] > 0.8:
        return "Very Negative"
    elif sentiment_score[0] > 0.6:
        return "Negative"
    elif sentiment_score[0] > 0.4:
        return "Neutral"
    elif sentiment_score[0] > 0.2:
        return "Positive"
    else:
        return "Very Positive"

def get_stock_data(stock_number):
    url = f"https://finance.yahoo.co.jp/quote/{stock_number}.T/history"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        
        price_elements = soup.select('td[class*="hPrNlgCj"]')  # Updated class name
        prices = [float(element.text.replace(',', '')) for element in price_elements if element.text.strip() != '']
        
        if not prices:
            print("Warning: No price data found. The webpage structure might have changed.")
            return None
        
        return prices[-30:]  # Return last 30 days of data
    except requests.RequestException as e:
        print(f"Error retrieving stock data: {e}")
        return None

def identify_pattern(prices):
    if prices is None or len(prices) < 5:
        return "Insufficient data for pattern identification"
    
    # Calculate price changes
    changes = np.diff(prices)
    
    # Identify trend
    if np.all(changes > 0):
        return "Upward Trend"
    elif np.all(changes < 0):
        return "Downward Trend"
    
    # Check for specific patterns
    n = len(prices)
    if n >= 5:
        if prices[n-1] > prices[n-2] > prices[n-3] < prices[n-4] < prices[n-5]:
            return "V-Shape Recovery"
        elif prices[n-1] < prices[n-2] < prices[n-3] > prices[n-4] > prices[n-5]:
            return "Inverted V-Shape"
    
    if n >= 7:
        if prices[n-1] > prices[n-4] and prices[n-4] < prices[n-7]:
            return "Double Bottom"
        elif prices[n-1] < prices[n-4] and prices[n-4] > prices[n-7]:
            return "Double Top"
    
    # If no specific pattern is found
    return "No Clear Pattern"

def main():
    while True:
        stock_number = input("Enter the stock exchange number (e.g., 3092 for ZOZO): ")
        company_name = get_company_name(stock_number)
        
        if company_name:
            confirmation = input(f"The company for stock number {stock_number} is {company_name}. Is this correct? (y/n): ")
            if confirmation.lower() == 'y':
                break
            else:
                print("Let's try again with a different stock number.")
        else:
            print("Company name could not be found. Please enter a valid stock exchange number.")

    purchase_price = float(input("Enter the price of the stock when you bought it (if not purchased, enter 0): "))

    tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")
    
    nikkei_news_data = scrape_nikkei_news(stock_number)
    yahoo_finance_news_data = scrape_yahoo_finance_news(stock_number)
    
    nikkei_sentiments = []
    yahoo_finance_sentiments = []
    
    for news in nikkei_news_data:
        title = news['title']
        sentiment = analyze_sentiment(title, tokenizer, model)
        nikkei_sentiments.append(sentiment)
    
    for news in yahoo_finance_news_data:
        title = news['title']
        sentiment = analyze_sentiment(title, tokenizer, model)
        yahoo_finance_sentiments.append(sentiment)
    
    # Nikkei overall sentiment analysis
    nikkei_sentiment_counts = {
        "Very Negative": nikkei_sentiments.count("Very Negative"),
        "Negative": nikkei_sentiments.count("Negative"),
        "Neutral": nikkei_sentiments.count("Neutral"),
        "Positive": nikkei_sentiments.count("Positive"),
        "Very Positive": nikkei_sentiments.count("Very Positive")
    }
    nikkei_overall_sentiment = max(nikkei_sentiment_counts, key=nikkei_sentiment_counts.get)
    
    # Yahoo Finance overall sentiment analysis
    yahoo_finance_sentiment_counts = {
        "Very Negative": yahoo_finance_sentiments.count("Very Negative"),
        "Negative": yahoo_finance_sentiments.count("Negative"),
        "Neutral": yahoo_finance_sentiments.count("Neutral"),
        "Positive": yahoo_finance_sentiments.count("Positive"),
        "Very Positive": yahoo_finance_sentiments.count("Very Positive")
    }
    yahoo_finance_overall_sentiment = max(yahoo_finance_sentiment_counts, key=yahoo_finance_sentiment_counts.get)
    
    # Overall sentiment analysis
    all_sentiments = nikkei_sentiments + yahoo_finance_sentiments
    overall_sentiment_counts = {
        "Very Negative": all_sentiments.count("Very Negative"),
        "Negative": all_sentiments.count("Negative"),
        "Neutral": all_sentiments.count("Neutral"),
        "Positive": all_sentiments.count("Positive"),
        "Very Positive": all_sentiments.count("Very Positive")
    }
    overall_sentiment = max(overall_sentiment_counts, key=overall_sentiment_counts.get)
    
    # Get stock data and identify pattern
    stock_prices = get_stock_data(stock_number)
    if stock_prices:
        matched_pattern = identify_pattern(stock_prices)
    else:
        matched_pattern = "Unable to retrieve stock data"
    
    # Get current stock price
    current_stock_price = get_current_stock_price(stock_number)
    
    if current_stock_price is not None:
        print(f"\nStock Analysis for {company_name} ({stock_number}):")
        print(f"Current Price: ¥{current_stock_price:.2f}")
        
        if purchase_price > 0:
            price_difference = current_stock_price - purchase_price
            price_percentage = (price_difference / purchase_price) * 100
            print(f"Purchase Price: ¥{purchase_price:.2f}")
            print(f"Price Difference: ¥{price_difference:.2f} ({price_percentage:.2f}%)")
        else:
            print("Purchase Price: N/A")
            print("Price Difference: N/A")
        
        print(f"\nNikkei Overall Sentiment: {nikkei_overall_sentiment}")
        print(f"Yahoo Finance Overall Sentiment: {yahoo_finance_overall_sentiment}")
        print(f"Overall Sentiment: {overall_sentiment}")
        print(f"Identified Pattern: {matched_pattern}")
        
        # Updated decision logic
        if matched_pattern == "Unable to retrieve stock data":
            if overall_sentiment in ["Very Positive", "Positive"]:
                decision = "Consider Buying (based on sentiment only)"
            elif overall_sentiment in ["Very Negative", "Negative"]:
                decision = "Consider Selling (based on sentiment only)"
            else:
                decision = "Hold (insufficient data for strong recommendation)"
        else:
            if overall_sentiment in ["Very Positive", "Positive"] and matched_pattern in ["Upward Trend", "V-Shape Recovery", "Double Bottom"]:
                decision = "Buy"
            elif overall_sentiment in ["Very Negative", "Negative"] and matched_pattern in ["Downward Trend", "Inverted V-Shape", "Double Top"]:
                decision = "Sell"
            else:
                decision = "Hold"
        
        print(f"\nRecommended Action: {decision}")
    else:
        print("Unable to retrieve current stock price. Please check the stock number and try again.")

if __name__ == '__main__':
    main()
