#fetch tickers from NASDAQ, AMEX and S&P500
#retrieves historical stock data using yfinance
import yfinance as yf
import os 
import pandas as pd 
import requests #make http requests
from datetime import datetime 
from bs4 import BeautifulSoup #web scraping html pages
from ftplib import FTP 


def get_stock_tickers():
    nasdaq_ftp_url = 'ftp.nasdaqtrader.com'
    nasdaq_file = '/SymbolDirectory/nasdaqlisted.txt'
    amex_file = '/SymbolDirectory/otherlisted.txt'
    
    
    def get_tickers_from_ftp(ftp_url, file_path):
        ftp = FTP(ftp_url)
        ftp.login()  
        tickers = []

        #download file from ftp and save locally 
        with open('temp.txt', 'wb') as f:
            ftp.retrbinary(f"RETR {file_path}", f.write)

        with open('temp.txt', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                ticker = line.split('|')[0].strip()
                if ticker and 'Y' not in line:
                    tickers.append(ticker)

        os.remove('temp.txt')  #remove temporary file
        return tickers

    #scrape s&p 500 tickers
    def get_sp500_tickers(): 
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url) #http request to wiki page
        soup = BeautifulSoup(response.text, 'html.parser') #parse html page
        table = soup.find('table', {'id': 'constituents'})  
        
        sp500_tickers = []
        for row in table.find_all('tr')[1:]:  
            ticker = row.find_all('td')[0].text.strip() #extract ticker
            sp500_tickers.append(ticker)
        
        print(f"Found {len(sp500_tickers)} S&P 500 tickers.")
        return sp500_tickers
    
    
    nasdaq_tickers = get_tickers_from_ftp(nasdaq_ftp_url, nasdaq_file)
    amex_tickers = get_tickers_from_ftp(nasdaq_ftp_url, amex_file)
    sp500_tickers = get_sp500_tickers()
    
    #combine tickers from exchanges and remove duplicates
    all_tickers = list(set(nasdaq_tickers + amex_tickers + sp500_tickers))
    print(f"Total tickers: {len(all_tickers)}")
    
    return all_tickers


def fetch_and_update_data(tickers, start_date):
    if not os.path.exists("../data"):
        os.makedirs("../data")
        
    for ticker in tickers:
        file_path = os.path.join("../data", f"{ticker}_data.csv")
        
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            last_date = existing_data.index[-1].strftime('%Y-%m-%d') #latest date
            
            new_data = yf.download(ticker, start=last_date, end=datetime.today().strftime('%Y-%m-%d'))
            
            if not new_data.empty:
                
                updated_data = pd.concat([existing_data, new_data])
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                updated_data.to_csv(file_path)
                print(f"Updated data for {ticker}")
            else:
                print(f"No new data to update for {ticker}")
        else:
            
            print(f"Fetching new data for {ticker}")
            data = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
            if not data.empty:
                data.to_csv(file_path)

if __name__ == "__main__":
    tickers = get_stock_tickers() #get tickers 
    start_date = '2020-01-01'  #define start date for data
    fetch_and_update_data(tickers, start_date) #fetch and update tickers
