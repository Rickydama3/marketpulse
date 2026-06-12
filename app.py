import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="MarketPulse")

st.title("📈 MarketPulse")
st.subheader("Stock Market Sentiment-Price Correlation Dashboard")

ticker = st.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
)

data = yf.download(
    ticker,
    start="2023-01-01",
    end="2025-01-01"
)

st.write("Recent Stock Data")
st.dataframe(data.tail())

st.line_chart(data["Close"])

st.success("Dashboard Working Successfully")