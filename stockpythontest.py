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

def scrape_toyo_keizai_news(stock_number):
    url = f"https://shikiho.toyokeizai.net/search?type=articles&query={stock_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.select('div.list__card__title a')
    news_data = []
    for item in news_items:
        title = item.text.strip()
        url = "https://shikiho.toyokeizai.net" + item['href']
        news_data.append({"title": title, "url": url})
    return news_data

def scrape_kabutan_news(stock_number):
    url = f"https://kabutan.jp/stock/news?code={stock_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.select('div.news_title a')
    news_data = []
    for item in news_items:
        title = item.text.strip()
        url = "https://kabutan.jp" + item['href']
        news_data.append({"title": title, "url": url})
    return news_data

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
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
    url = f"https://finance.yahoo.co.jp/quote/{stock_number}.T/history"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try different class names
        price_elements = soup.select('td[class*="hPrNlgCj"], td[class*="price"]')
        
        if not price_elements:
            print("Warning: Could not find price elements. Trying an alternative method.")
            # Alternative method: look for any td with a numeric content
            price_elements = [td for td in soup.find_all('td') if td.text.strip().replace(',', '').isdigit()]
        
        prices = []
        for element in price_elements:
            try:
                price = float(element.text.strip().replace(',', ''))
                prices.append(price)
            except ValueError:
                continue  # Skip non-numeric values
        
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
    
    changes = np.diff(prices)
    
    if np.all(changes > 0):
        return "Upward Trend"
    elif np.all(changes < 0):
        return "Downward Trend"
    
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
    toyo_keizai_news_data = scrape_toyo_keizai_news(stock_number)
    kabutan_news_data = scrape_kabutan_news(stock_number)
    
    news_sources = {
        "Nikkei": nikkei_news_data,
        "Yahoo Finance": yahoo_finance_news_data,
        "Toyo Keizai": toyo_keizai_news_data,
        "Kabutan": kabutan_news_data
    }
    
    sentiment_data = {}
    for source, news_data in news_sources.items():
        sentiments = []
        for news in news_data:
            sentiment_score = analyze_sentiment(news['title'], tokenizer, model)
            sentiments.append({
                "title": news['title'],
                "url": news['url'],
                "sentiment": sentiment_score,
                "sentiment_text": sentiment_to_text(sentiment_score)
            })
        sentiment_data[source] = sentiments
    
    def get_overall_sentiment(sentiments):
        if not sentiments:
            return "No data"
        average_sentiment = sum(s['sentiment'] for s in sentiments) / len(sentiments)
        return sentiment_to_text(average_sentiment)
    
    overall_sentiments = {source: get_overall_sentiment(sentiments) for source, sentiments in sentiment_data.items()}
    all_sentiments = [s['sentiment'] for sentiments in sentiment_data.values() for s in sentiments]
    overall_sentiment = sentiment_to_text(sum(all_sentiments) / len(all_sentiments)) if all_sentiments else "No data"
    
    stock_prices = get_stock_data(stock_number)
    if stock_prices:
        matched_pattern = identify_pattern(stock_prices)
    else:
        matched_pattern = "Unable to retrieve historical data"
    
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
        
        for source, sentiment in overall_sentiments.items():
            print(f"{source} Overall Sentiment: {sentiment}")
        print(f"Overall Sentiment: {overall_sentiment}")
        print(f"Identified Pattern: {matched_pattern}")
        
        if matched_pattern == "Unable to retrieve historical data":
            print("Note: Pattern identification is based on current price only due to lack of historical data.")
            if current_stock_price > purchase_price:
                matched_pattern = "Price has increased since purchase"
            elif current_stock_price < purchase_price:
                matched_pattern = "Price has decreased since purchase"
            else:
                matched_pattern = "Price unchanged since purchase"
        
        if overall_sentiment in ["Very Positive", "Positive"] and matched_pattern in ["Upward Trend", "V-Shape Recovery", "Double Bottom", "Price has increased since purchase"]:
            decision = "Buy"
        elif overall_sentiment in ["Very Negative", "Negative"] and matched_pattern in ["Downward Trend", "Inverted V-Shape", "Double Top", "Price has decreased since purchase"]:
            decision = "Sell"
        else:
            decision = "Hold"
        
        print(f"\nRecommended Action: {decision}")
        
        while True:
            source_request = input("\nEnter a news source name to see detailed sentiment analysis (or 'quit' to exit): ")
            if source_request.lower() == 'quit':
                break
            
            if source_request in news_sources:
                print(f"\nDetailed Sentiment Analysis for {source_request}:")
                for article in sentiment_data[source_request]:
                    print(f"Title: {article['title']}")
                    print(f"Sentiment: {article['sentiment_text']}")
                    print(f"URL: {article['url']}")
                    print()
            else:
                print("Invalid source name. Available sources: Nikkei, Yahoo Finance, Toyo Keizai, Kabutan")
    else:
        print("Unable to retrieve current stock price. Please check the stock number and try again.")

if __name__ == '__main__':
    main()
