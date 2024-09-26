import requests
import pandas as pd
from newsapi import NewsApiClient
import datetime
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

load_dotenv()

api_key = os.getenv("API_KEY")

newsapi = NewsApiClient(api_key=api_key)

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone') #pre-trained finbert model for sentiment analysis 
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

finbert_sentiment = pipeline('sentiment analysis', model=model, tokenizer=tokenizer)

def fetch_news(ticker, from_date=None, to_date=None):
    if not from_date:
        from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    
    if not to_date:
        to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Fetch news articles
    articles = newsapi.get_everything(q=ticker,
                                    from_param=from_date,
                                    to=to_date,
                                    language='en',
                                    sort_by='relevancy')
    
    news_data = []
    for article in articles['articles']:
        title = article['title']
        description = article['description']
        
        sentiment = finbert_sentiment(title + " " + description if description else title)
        sentiment_label = sentiment[0]['label']
        
        news_data.append({
            'source': article['source']['name'],
            'title': article['title'],
            'description': article['description'],
            'sentiment': sentiment_label,
            'url': article['url'],
            'published_at': article['publishedAt']
        })
    
    
    df = pd.DataFrame(news_data)
    return df

if __name__ == "__main__":
    df = fetch_news(ticker="AAPL")
    print(df.head())
    df.to_csv('../data/news_aapl.csv', index=False)