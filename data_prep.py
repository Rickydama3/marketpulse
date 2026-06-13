"""
MarketPulse - Data Preparation
Fetches stock price history and generates/loads news headline sentiment scores.
"""

import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
           "NVDA", "META", "NFLX", "JPM", "DIS"]

def fetch_price_data(ticker, period="6mo"):
    """Download historical OHLCV data for a ticker."""
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Ticker"] = ticker
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

def load_headlines(path="headlines.csv"):
    """
    Load headlines CSV with columns: Date, Ticker, Headline
    You can populate this manually or scrape from a news API.
    """
    return pd.read_csv(path, parse_dates=["Date"])

def score_sentiment(headlines_df):
    """Add VADER compound sentiment score for each headline."""
    sia = SentimentIntensityAnalyzer()
    headlines_df["sentiment"] = headlines_df["Headline"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    # Aggregate to daily average sentiment per ticker
    daily_sentiment = (
        headlines_df.groupby(["Date", "Ticker"])["sentiment"]
        .mean()
        .reset_index()
    )
    return daily_sentiment

def build_dataset(tickers=TICKERS, headlines_path="headlines.csv"):
    """Merge price data with sentiment data for all tickers."""
    all_price = []
    for t in tickers:
        try:
            all_price.append(fetch_price_data(t))
        except Exception as e:
            print(f"Failed to fetch {t}: {e}")

    price_df = pd.concat(all_price, ignore_index=True)

    headlines_df = load_headlines(headlines_path)
    sentiment_df = score_sentiment(headlines_df)

    merged = price_df.merge(sentiment_df, on=["Date", "Ticker"], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0)

    return merged

if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("merged_data.csv", index=False)
    print(f"Saved merged_data.csv with {len(df)} rows")