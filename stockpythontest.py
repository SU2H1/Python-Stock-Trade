import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime
import json
from langdetect import detect

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

def scrape_google_finance_news(stock_number):
    url = f"https://www.google.com/finance/quote/{stock_number}:TYO?window=1M"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('a', href=lambda href: href and "/finance/" in href)
    news_data = []
    for item in news_items:
        title = item.text.strip()
        url = "https://www.google.com" + item['href']
        news_data.append({"title": title, "url": url})
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
    url = f"https://www.google.com/finance/quote/{stock_number}:TYO?window=1M"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Find the script tag containing the stock data
        soup = BeautifulSoup(response.content, 'html.parser')
        scripts = soup.find_all('script')
        
        data_script = None
        for script in scripts:
            if 'var data = ' in script.text:
                data_script = script
                break
        
        if data_script is None:
            print("Could not find stock data in the page.")
            return None
        
        # Extract the JSON data
        json_data = data_script.text.split('var data = ')[1].split(';')[0]
        data = json.loads(json_data)
        
        # Extract the stock price data
        stock_data = []
        for entry in data[1:]:  # Skip the first entry as it's usually the column headers
            timestamp = entry[0]['timestamp']
            date = datetime.fromtimestamp(timestamp)
            price = entry[1]
            stock_data.append((date, price))
        
        # Sort by date (newest first) and return up to 30 days of data
        stock_data.sort(key=lambda x: x[0], reverse=True)
        return stock_data[:30]
    
    except requests.RequestException as e:
        print(f"Error retrieving stock data: {e}")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing stock data: {e}")
        return None

def identify_pattern(stock_data):
    if stock_data is None or len(stock_data) < 5:
        return "Insufficient data for pattern identification"
    
    prices = [price for _, price in stock_data]
    changes = np.diff(prices)
    
    if np.all(changes <= 0):
        return "Downward Trend"
    elif np.all(changes >= 0):
        return "Upward Trend"
    
    n = len(prices)
    if n >= 5:
        if prices[0] > prices[1] > prices[2] < prices[3] < prices[4]:
            return "V-Shape Recovery"
        elif prices[0] < prices[1] < prices[2] > prices[3] > prices[4]:
            return "Inverted V-Shape"
    
    if n >= 7:
        if prices[0] > prices[3] and prices[3] < prices[6]:
            return "Double Bottom"
        elif prices[0] < prices[3] and prices[3] > prices[6]:
            return "Double Top"
    
    return "No Clear Pattern"

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
        google_finance_news_data = scrape_google_finance_news(stock_number)
        
        nikkei_sentiments = [analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model) for news in nikkei_news_data]
        yahoo_finance_sentiments = [analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model) for news in yahoo_finance_news_data]
        google_finance_sentiments = [analyze_sentiment(news['title'], ja_tokenizer, ja_model, en_tokenizer, en_model) for news in google_finance_news_data]
        
        def get_overall_sentiment(sentiments):
            if not sentiments:
                return "No data"
            avg_sentiment = sum(sentiments) / len(sentiments)
            return sentiment_to_text(avg_sentiment)
        
        nikkei_overall_sentiment = get_overall_sentiment(nikkei_sentiments)
        yahoo_finance_overall_sentiment = get_overall_sentiment(yahoo_finance_sentiments)
        google_finance_overall_sentiment = get_overall_sentiment(google_finance_sentiments)
        
        # Calculate overall sentiment as (Yahoo article value + Nikkei article rating value + Google article value) / 3
        if nikkei_sentiments and yahoo_finance_sentiments and google_finance_sentiments:
            overall_sentiment_value = (sum(nikkei_sentiments) / len(nikkei_sentiments) + sum(yahoo_finance_sentiments) / len(yahoo_finance_sentiments) + sum(google_finance_sentiments) / len(google_finance_sentiments)) / 3
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
            
            print("\n30-Day Stock Price Analysis:")
            if stock_data:
                for date, price in stock_data:
                    print(f"{date.strftime('%Y-%m-%d')}: ¥{price:.2f}")
            else:
                print("Unable to retrieve 30-day stock data")
            
            print(f"\nIdentified 30-Day Pattern: {matched_pattern}")
            
            print(f"\nNikkei Overall Sentiment: {nikkei_overall_sentiment}")
            print(f"Yahoo Finance Overall Sentiment: {yahoo_finance_overall_sentiment}")
            print(f"Google Finance Overall Sentiment: {google_finance_overall_sentiment}")
            print(f"Overall Sentiment: {overall_sentiment}")
            
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
            
            while True:
                source_request = input("\nEnter a news source name to see detailed sentiment analysis (or 'quit' to exit): ")
                if source_request.lower() == 'quit':
                    break
                
                news_sources = {
                    "Nikkei": list(zip(nikkei_news_data, map(sentiment_to_text, nikkei_sentiments))),
                    "Yahoo Finance": list(zip(yahoo_finance_news_data, map(sentiment_to_text, yahoo_finance_sentiments))),
                    "Google Finance": list(zip(google_finance_news_data, map(sentiment_to_text, google_finance_sentiments)))
                }
                
                if source_request in news_sources:
                    print(f"\nDetailed Sentiment Analysis for {source_request}:")
                    for article, sentiment in news_sources[source_request]:
                        print(f"Title: {article['title']}")
                        print(f"Sentiment: {sentiment}")
                        print(f"URL: {article['url']}")
                        print()
                else:
                    print("Invalid source name. Available sources: Nikkei, Yahoo Finance, Google Finance")
        else:
            print("Unable to retrieve current stock price. Please check the stock number and try again.")
    except Exception as e:
        print(f"An error occurred while processing the stock data: {str(e)}")
        print("Please check the stock number and try again.")

if __name__ == '__main__':
    main()
