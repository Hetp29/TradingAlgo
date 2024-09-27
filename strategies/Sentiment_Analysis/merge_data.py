import pandas as pd

def merge_news_and_stock(news_file, stock_file):
    # Load the CSV files
    news_df = pd.read_csv(news_file)
    stock_df = pd.read_csv(stock_file)

    # Convert 'published_at' in news_df to datetime and remove timezone
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce').dt.tz_localize(None)
    
    # Convert 'Date' in stock_df to datetime
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')

    # Verify the dtypes after conversion
    print(news_df.dtypes)
    print(stock_df.dtypes)

    # Merge the news_df with stock_df based on published_at and Date
    merged_df = pd.merge_asof(
        news_df.sort_values('published_at'),
        stock_df.sort_values('Date'),
        left_on='published_at',
        right_on='Date',
        direction='forward'
    )  # Merge with the next available stock date if news was after market hours

    return merged_df

if __name__ == "__main__":
    news_file = '../../output/news_aapl.csv'
    stock_file = '../../data/AAPL_data.csv'

    merged_df = merge_news_and_stock(news_file, stock_file)
    print(merged_df.head())

    # Save the merged data
    merged_df.to_csv('../../output/merged_news_stock_aapl.csv', index=False)
