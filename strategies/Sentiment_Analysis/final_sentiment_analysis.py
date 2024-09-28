import pandas as pd
import matplotlib.pyplot as plt

def merge_news_and_stock(news_file, stock_file):
    
    news_df = pd.read_csv(news_file)
    stock_df = pd.read_csv(stock_file)

    
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce').dt.tz_localize(None)
    
    
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')


    merged_df = pd.merge_asof(
        news_df.sort_values('published_at'),
        stock_df.sort_values('Date'),
        left_on='published_at',
        right_on='Date',
        direction='forward'
    )
    
    
    merged_df['price_change'] = merged_df['Close'].pct_change() * 100

    
    def classify_price_movement(change):
        if change > 0.5:
            return 'Positive'
        elif change < -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    
    merged_df['price_movement'] = merged_df['price_change'].apply(classify_price_movement)

    return merged_df

def analyze_sentiment_vs_price_movement(merged_df):
    
    sentiment_vs_movement = merged_df.groupby(['sentiment', 'price_movement']).size().unstack(fill_value=0)
    print(sentiment_vs_movement)

    
    sentiment_vs_movement.plot(kind='bar', stacked=True)
    plt.title("Sentiment vs Stock Price Movement")
    plt.ylabel("Number of News Articles")
    plt.xlabel("Sentiment")
    plt.show()

if __name__ == "__main__":

    news_file = '../../output/news_aapl.csv'
    stock_file = '../../data/AAPL_data.csv'
    
    
    merged_df = merge_news_and_stock(news_file, stock_file)
    
    
    analyze_sentiment_vs_price_movement(merged_df)
    
    
    merged_df.to_csv('../../output/final_sentiment_analysis_aapl.csv', index=False)

