import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import date, timedelta

st.set_page_config(page_title="AI ML Stock Market Analyzer", layout="wide")

st.title("AI ML Stock Market Analyzer")
st.markdown("Analyze stock trends, technical indicators, risk, and a simple next day prediction.")

def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_span=12, long_span=26, signal_span=9):
    ema_short = series.ewm(span=short_span, adjust=False).mean()
    ema_long = series.ewm(span=long_span, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_risk_metrics(close_prices):
    returns = close_prices.pct_change().dropna()
    avg_daily_return = returns.mean()
    daily_volatility = returns.std()
    annual_return = avg_daily_return * 252
    annual_volatility = daily_volatility * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
    return {
        "Average Daily Return": avg_daily_return,
        "Daily Volatility": daily_volatility,
        "Expected Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio
    }

def add_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["Target"] = df["Close"].shift(-1)
    df["RSI"] = calculate_rsi(df["Close"])
    macd, signal, hist = calculate_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    return df.dropna().copy()

def train_model(df):
    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "MA_10", "MA_20", "MA_50",
        "Volatility_10", "RSI", "MACD", "MACD_Signal"
    ]
    model_df = df[feature_cols + ["Target"]].dropna().copy()

    if len(model_df) < 60:
        return None

    split_index = int(len(model_df) * 0.8)

    train_data = model_df.iloc[:split_index]
    test_data = model_df.iloc[split_index:]

    X_train = train_data[feature_cols]
    y_train = train_data["Target"]

    X_test = test_data[feature_cols]
    y_test = test_data["Target"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, feature_cols, X_test, y_test, predictions, mae, r2

def make_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], mode="lines", name="20 Day MA"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], mode="lines", name="50 Day MA"))
    fig.update_layout(title=f"{ticker} Price Trend", xaxis_title="Date", yaxis_title="Price")
    return fig

def make_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(y=70)
    fig.add_hline(y=30)
    fig.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    return fig

def make_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram"))
    fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value")
    return fig

def prediction_chart(actual_index, actual_values, predicted_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_index, y=actual_values, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=actual_index, y=predicted_values, mode="lines", name="Predicted"))
    fig.update_layout(title="Model Prediction vs Actual", xaxis_title="Date", yaxis_title="Next Day Close")
    return fig

st.sidebar.header("Input Settings")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
end_date = date.today()
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365 * 2))
selected_end_date = st.sidebar.date_input("End Date", end_date)

if ticker:
    try:
        raw_df = load_data(ticker, start_date, selected_end_date)

        if raw_df.empty:
            st.error("No data found. Please check the ticker symbol or date range.")
        else:
            df = add_features(raw_df)

            st.subheader(f"Analyzed Data for {ticker}")
            st.dataframe(df.tail(15), use_container_width=True)

            latest_close = df["Close"].iloc[-1]
            latest_rsi = df["RSI"].iloc[-1]
            latest_volume = df["Volume"].iloc[-1] if "Volume" in df.columns else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Close", f"{latest_close:.2f}")
            c2.metric("Latest RSI", f"{latest_rsi:.2f}")
            c3.metric("Latest Volume", f"{latest_volume:,.0f}")

            st.plotly_chart(make_price_chart(df, ticker), use_container_width=True)

            left, right = st.columns(2)

            with left:
                st.plotly_chart(make_rsi_chart(df), use_container_width=True)

            with right:
                st.plotly_chart(make_macd_chart(df), use_container_width=True)

            st.subheader("Risk Metrics")

            metrics = calculate_risk_metrics(df["Close"])
            metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            metrics_df["Value"] = metrics_df["Value"].apply(lambda x: f"{x:.6f}")

            st.table(metrics_df)

            st.subheader("AI ML Prediction")

            result = train_model(df)

            if result is None:
                st.warning("Not enough data to train the model. Please select a larger date range.")
            else:
                model, feature_cols, X_test, y_test, predictions, mae, r2 = result

                st.write(f"Model Used: Linear Regression")
                st.write(f"Mean Absolute Error: {mae:.4f}")
                st.write(f"R2 Score: {r2:.4f}")

                st.plotly_chart(
                    prediction_chart(y_test.index, y_test.values, predictions),
                    use_container_width=True
                )

                latest_features = df[feature_cols].iloc[[-1]]
                next_day_prediction = model.predict(latest_features)[0]

                st.success(f"Predicted next closing price for {ticker}: {next_day_prediction:.2f}")

            csv = df.to_csv().encode("utf-8")

            st.download_button(
                label="Download analyzed data as CSV",
                data=csv,
                file_name=f"{ticker}_analyzed_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")

        