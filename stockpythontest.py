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

# Scraping news from Nikkei
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

# Scraping news from Yahoo Finance
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

# Sentiment analysis
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

# Matching stock trends to patterns
def match_pattern(stock_data, patterns):
    # Placeholder function: in reality, this would involve more complex analysis
    # For simplicity, let's just return a dummy pattern
    return "Head & Shoulders"

# Fetch current stock price
def get_current_stock_price(stock_number):
    url = f"https://finance.yahoo.co.jp/quote/{stock_number}.T"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    price_tag = soup.find('span', class_='_3rXUuGXGCJdGmsrlhlR34E')
    if price_tag:
        return float(price_tag.text.replace(',', ''))
    else:
        return None

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
    
    # Placeholder stock trend data
    stock_data = np.random.rand(100)  # Dummy data for example
    pattern_match = match_pattern(stock_data, ["Head & Shoulders", "Wedge", "Symmetrical triangle", "Ascending triangle", "Double bottom", "Pennant", "Triple bottom", "Price channel", "Cup and handle", "Rounding bottom", "Flag", "Triple top", "GAPS"])
    
    # Get current stock price
    current_stock_price = get_current_stock_price(stock_number)
    
    print(f"Nikkei Overall Sentiment: {nikkei_overall_sentiment}")
    print(f"Yahoo Finance Overall Sentiment: {yahoo_finance_overall_sentiment}")
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Matched Pattern: {pattern_match}")
    print(f"Current Stock Price: {current_stock_price}")
    
    # Decision logic based on sentiment and pattern
    if overall_sentiment in ["Very Positive", "Positive"] and pattern_match in ["Double bottom", "Rounding bottom", "Cup and handle"]:
        decision = "Buy"
    elif overall_sentiment in ["Very Negative", "Negative"] and pattern_match in ["Head & Shoulders", "Triple top"]:
        decision = "Sell"
    else:
        decision = "Hold"
    
    print(f"Decision: {decision}")

if __name__ == '__main__':
    main()
