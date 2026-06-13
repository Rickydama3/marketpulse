# MarketPulse — Stock Sentiment-Price Correlation Dashboard

Analyzes correlation between news sentiment and stock price movements,
trains a classifier to predict next-day price direction, and visualizes
everything in an interactive Streamlit dashboard.

## Setup

```bash
pip install yfinance nltk scikit-learn pandas streamlit matplotlib joblib
```

## How to Run

1. **Prepare data** (fetches price history + scores headline sentiment):
   ```bash
   python data_prep.py
   ```
   This creates `merged_data.csv`. Edit `headlines.csv` to add more
   headlines (Date, Ticker, Headline columns) for richer sentiment signal.

2. **Train the model**:
   ```bash
   python train_model.py
   ```
   This creates `model.pkl` and `backtest_results.csv`, and prints
   accuracy + Sharpe ratio metrics.

3. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `data_prep.py` — fetches OHLCV data via yfinance, scores headlines with
  VADER sentiment, merges into `merged_data.csv`
- `headlines.csv` — sample news headlines (replace/expand with real data
  from a news API for better results)
- `train_model.py` — feature engineering (rolling averages, volatility,
  lagged sentiment) + RandomForest classifier + backtesting with Sharpe ratio
- `app.py` — Streamlit dashboard: price charts, sentiment bars,
  predictions table, and strategy vs buy-and-hold backtest plot

## Notes for Resume / Interview

- Talk through: why VADER (fast, no training needed, decent for short text),
  why RandomForest (handles non-linear feature interactions, robust to
  small datasets), and why `shuffle=False` in train/test split (avoids
  lookahead bias in time-series).
- To improve: swap VADER for a transformer-based sentiment model
  (e.g. FinBERT), add more tickers/headlines, try XGBoost or LSTM.
- The provided `headlines.csv` is a small sample — for real numbers,
  scrape headlines from a news API (e.g. NewsAPI, Finnhub) matching the
  price data's date range.