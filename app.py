"""
MarketPulse - Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from train_model import engineer_features, FEATURE_COLS, backtest

st.set_page_config(page_title="MarketPulse", layout="wide")
st.title("📈 MarketPulse — Sentiment vs Price Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_data.csv", parse_dates=["Date"])
    return engineer_features(df)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

tickers = sorted(df["Ticker"].unique())
ticker = st.sidebar.selectbox("Select Ticker", tickers)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["Date"].min(), df["Date"].max())
)

sub = df[df["Ticker"] == ticker]
if len(date_range) == 2:
    sub = sub[(sub["Date"] >= pd.Timestamp(date_range[0])) &
              (sub["Date"] <= pd.Timestamp(date_range[1]))]

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{ticker} — Price Trend")
    fig, ax = plt.subplots()
    ax.plot(sub["Date"], sub["Close"], label="Close Price")
    ax.plot(sub["Date"], sub["ma_5"], label="5-day MA", linestyle="--")
    ax.legend()
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)

with col2:
    st.subheader(f"{ticker} — Sentiment Score")
    fig2, ax2 = plt.subplots()
    ax2.bar(sub["Date"], sub["sentiment"], color="orange")
    ax2.set_ylabel("VADER Sentiment")
    st.pyplot(fig2)

st.subheader("Model Predictions (Next-Day Direction)")
sub_disp = sub[["Date", "Close", "sentiment", "target"]].copy()
sub_disp["predicted"] = model.predict(sub[FEATURE_COLS])
sub_disp["predicted"] = sub_disp["predicted"].map({1: "📈 Up", 0: "📉 Down"})
sub_disp["target"] = sub_disp["target"].map({1: "📈 Up", 0: "📉 Down"})
st.dataframe(sub_disp.tail(15), use_container_width=True)

st.subheader("Backtest: Strategy vs Buy & Hold")
bt_df, cum_strat, cum_bh = backtest(df[df["Ticker"] == ticker], model)
fig3, ax3 = plt.subplots()
ax3.plot(bt_df["Date"], cum_strat.values, label="Sentiment Strategy")
ax3.plot(bt_df["Date"], cum_bh.values, label="Buy & Hold")
ax3.legend()
ax3.set_ylabel("Cumulative Return")
st.pyplot(fig3)

st.caption("Built with yfinance, NLTK VADER, scikit-learn, and Streamlit.")