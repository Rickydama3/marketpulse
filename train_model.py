"""
MarketPulse - Model Training
Engineers time-series + sentiment features and trains a classifier
to predict next-day price direction (up/down).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def engineer_features(df):
    """Build rolling averages, lagged sentiment, returns, and target labels."""
    df = df.sort_values(["Ticker", "Date"]).copy()

    features = []
    for ticker, g in df.groupby("Ticker"):
        g = g.copy()
        g["return"] = g["Close"].pct_change()
        g["ma_5"] = g["Close"].rolling(5).mean()
        g["ma_10"] = g["Close"].rolling(10).mean()
        g["volatility_5"] = g["return"].rolling(5).std()
        g["sentiment_lag1"] = g["sentiment"].shift(1)
        g["sentiment_ma3"] = g["sentiment"].rolling(3).mean()

        # Target: 1 if next day's close is higher than today's close
        g["target"] = (g["Close"].shift(-1) > g["Close"]).astype(int)

        features.append(g)

    full = pd.concat(features, ignore_index=True)
    full = full.dropna()
    return full

FEATURE_COLS = ["return", "ma_5", "ma_10", "volatility_5",
                "sentiment", "sentiment_lag1", "sentiment_ma3"]

def train_model(df):
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # preserve time order
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds))

    return model, acc

def backtest(df, model):
    """Simple backtest: go long when model predicts 'up', compute Sharpe ratio."""
    X = df[FEATURE_COLS]
    df = df.copy()
    df["pred"] = model.predict(X)

    # Strategy return: if predicted up, capture next-day return; else 0
    df["strategy_return"] = np.where(df["pred"] == 1, df["return"].shift(-1), 0)
    df["buy_hold_return"] = df["return"].shift(-1)

    df = df.dropna(subset=["strategy_return", "buy_hold_return"])

    def sharpe(returns, periods_per_year=252):
        return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

    strat_sharpe = sharpe(df["strategy_return"])
    bh_sharpe = sharpe(df["buy_hold_return"])

    print(f"Strategy Sharpe Ratio: {strat_sharpe:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {bh_sharpe:.2f}")

    cum_strategy = (1 + df["strategy_return"]).cumprod()
    cum_buyhold = (1 + df["buy_hold_return"]).cumprod()

    return df, cum_strategy, cum_buyhold

if __name__ == "__main__":
    raw = pd.read_csv("merged_data.csv", parse_dates=["Date"])
    feat_df = engineer_features(raw)

    model, acc = train_model(feat_df)
    joblib.dump(model, "model.pkl")

    bt_df, cum_strat, cum_bh = backtest(feat_df, model)
    bt_df.to_csv("backtest_results.csv", index=False)

    print("Saved model.pkl and backtest_results.csv")