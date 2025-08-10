import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threading import RLock
import time
import gc
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Thread lock for plotting safety
_lock = RLock()

st.set_page_config(page_title="Stock Trend Prediction with Sentiment", layout="wide")
st.title("📈 Stock Trend Prediction with Sentiment Analysis (Demo)")

@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*"
    })
    return s

@st.cache_resource
def get_sia():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

session = get_session()
sia = get_sia()


def safe_json_from_response(resp):
    """Return (json, error_message). Handles empty/HTML responses."""
    if resp is None:
        return None, "No response object."
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code} - {resp.reason}"
    text = resp.text or ""
    if not text.strip():
        return None, "Empty response body."
    try:
        return resp.json(), None
    except Exception:
        sample = text[:400].replace('\n', ' ')
        return None, f"Invalid JSON. Response starts with: {sample!r}"


@st.cache_data(ttl=3600)
def find_ticker_by_name(name):
    """Search Yahoo Finance for a ticker symbol matching name. Returns (symbol, message)."""
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        resp = session.get(url, params={'q': name}, timeout=10)
        data, err = safe_json_from_response(resp)
        if err:
            return None, f"Search failed: {err}"
        quotes = data.get("quotes") or []
        if not quotes:
            return None, "No tickers found."
        for q in quotes:
            sym = q.get("symbol")
            if sym and sym.endswith(".NS"):
                return sym, None
        return quotes[0].get("symbol"), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def fetch_price_data_yf(ticker, period="6mo", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, actions=False)
        if hist is None or hist.empty:
            return None, "No historical price data found for this ticker."
        df = hist.rename(columns={'Close': 'price'})[['price']].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df, None
    except Exception as e:
        return None, f"Error fetching price data: {e}"

@st.cache_data(ttl=300)
def fetch_news_yf(ticker, max_articles=6):
    try:
        t = yf.Ticker(ticker)
        raw_news = getattr(t, "news", []) or []
        if not raw_news:
            return [], "No news found."
        news_list = []
        for n in raw_news[:max_articles]:
            news_list.append({
                "title": n.get('title', ''),
                "link": n.get('link', ''),
                "summary": n.get('summary') or '',
                "publisher": n.get('publisher') or '',
                "time": n.get('providerPublishTime')
            })
        return news_list, None
    except Exception as e:
        return [], f"Error fetching news: {e}"

def extract_text_from_url(url, max_chars=1000):
    try:
        resp = session.get(url, timeout=8)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs[:10])
        return text[:max_chars]
    except Exception:
        return ""

def analyze_sentiment_text(text):
    if not sia or not text:
        return "neutral", 0.0
    try:
        scores = sia.polarity_scores(str(text))
        comp = scores.get("compound", 0.0)
        if comp >= 0.05:
            return "positive", comp
        elif comp <= -0.05:
            return "negative", comp
        else:
            return "neutral", comp
    except Exception:
        return "neutral", 0.0

def create_time_features(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['second'] = df.index.second
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    return df

def add_lag_features(df):
    df = df.copy()
    if 'price' not in df.columns:
        return df
    df['lag1'] = df['price'].shift(1)
    df['lag2'] = df['price'].shift(2)
    df['lag3'] = df['price'].shift(3)
    df['rolling_mean_5'] = df['price'].rolling(window=5, min_periods=1).mean()
    df['rolling_std_5'] = df['price'].rolling(window=5, min_periods=1).std().fillna(0)
    df['price_change'] = df['price'].pct_change().fillna(0)
    return df

def train_xgb_model(train_df, features, target='price'):
    X = train_df[features].copy()
    y = train_df[target].copy()
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model

def plot_price(df, title="Price"):
    with _lock:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['price'], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

def plot_sentiment_counts(sentiment_series):
    with _lock:
        counts = sentiment_series.value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        counts.plot(kind='bar', ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig

def main():
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Input mode", ["Ticker (recommended)", "Company Name (auto-search)"])
    ticker_input = ""
    company_input = ""
    if mode == "Ticker (recommended)":
        ticker_input = st.sidebar.text_input("Enter ticker", value="TATAMOTORS.NS")
    else:
        company_input = st.sidebar.text_input("Enter company name", value="Tata Motors")
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    enable_model = st.sidebar.checkbox("Train XGBoost model (requires >=100 points)", value=True)
    max_articles = st.sidebar.slider("Max news articles to fetch", 1, 10, 5)  # <-- fixed scope here
    analyze_btn = st.sidebar.button("🔍 Analyze Stock")

    if analyze_btn:
        with st.spinner("Fetching data..."):
            if mode.startswith("Ticker"):
                ticker = ticker_input.strip()
                if not ticker:
                    st.error("Please enter a ticker symbol.")
                    return
            else:
                ticker, msg = find_ticker_by_name(company_input)
                if not ticker:
                    st.error(f"Could not find ticker: {msg}")
                    return
                st.success(f"Resolved ticker: {ticker}")

            df, msg = fetch_price_data_yf(ticker, period=period, interval=interval)
            if df is None:
                st.error(msg)
                return

            st.header("📈 Stock Data Overview")
            st.write(f"**Ticker:** {ticker} — **Period:** {period} — **Interval:** {interval}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Data Points", len(df))
            col2.metric("Current Price", f"₹{df['price'].iloc[-1]:.2f}")
            col3.metric("Price Change", f"₹{(df['price'].iloc[-1] - df['price'].iloc[-2]) if len(df)>1 else 0:.2f}")
            col4.metric("Volatility (std)", f"{df['price'].std():.2f}")

            csv = df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name=f"{ticker}_history.csv", mime="text/csv")

            st.subheader("Price Chart")
            fig_price = plot_price(df, title=f"{ticker} Price")
            st.pyplot(fig_price)
            plt.close(fig_price)

            if enable_model:
                if len(df) < 30:
                    st.warning("Not enough points for modeling.")
                else:
                    proc = create_time_features(df)
                    proc = add_lag_features(proc)
                    proc = proc.dropna()
                    if len(proc) < 30:
                        st.warning("Too few rows after feature creation.")
                    else:
                        split_idx = int(len(proc) * 0.8)
                        train = proc.iloc[:split_idx]
                        test = proc.iloc[split_idx:]
                        FEATURES = ['second','minute','hour','day','weekday','lag1','lag2','lag3','rolling_mean_5','rolling_std_5','price_change']
                        FEATURES = [f for f in FEATURES if f in train.columns]
                        if len(FEATURES) < 5:
                            st.warning("Insufficient features.")
                        else:
                            try:
                                model = train_xgb_model(train, FEATURES)
                                preds = model.predict(test[FEATURES])
                                rmse = (mean_squared_error(test['price'], preds))**0.5
                                st.subheader("Model Results")
                                c1, c2 = st.columns(2)
                                c1.metric("Test RMSE", f"{rmse:.2f}")
                                c2.metric("Train/Test size", f"{len(train)}/{len(test)}")
                                pred_df = test.copy()
                                pred_df['pred'] = preds
                                fig, ax = plt.subplots(figsize=(10,5))
                                ax.plot(train.index, train['price'], label='Train')
                                ax.plot(test.index, test['price'], label='Actual', color='orange')
                                ax.plot(test.index, preds, label='Predicted', linestyle='--', color='red')
                                ax.legend()
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                fi = model.feature_importances_
                                fi_df = pd.DataFrame({"feature": FEATURES, "importance": fi}).sort_values("importance", ascending=False)
                                st.subheader("Feature Importance")
                                st.table(fi_df.reset_index(drop=True))
                            except Exception as e:
                                st.error(f"Model training failed: {e}")

            st.header("📰 Sentiment Analysis (News Titles)")
            news_list, news_err = fetch_news_yf(ticker, max_articles)
            if news_err:
                st.warning(f"News fetch: {news_err}")

            if news_list:
                news_rows = []
                for n in news_list:
                    text = (n.get('title') or '') + " " + (n.get('summary') or '')
                    if not n.get('summary') and n.get('link'):
                        snippet = extract_text_from_url(n['link'], max_chars=400)
                        text += " " + snippet
                    label, score = analyze_sentiment_text(text)
                    news_rows.append({
                        "title": n.get('title'),
                        "publisher": n.get('publisher'),
                        "sentiment": label,
                        "score": float(score),
                        "link": n.get('link')
                    })
                news_df = pd.DataFrame(news_rows)
                for idx, row in news_df.iterrows():
                    with st.expander(f"{row['title']} — {row['publisher']}"):
                        st.write(f"**Sentiment:** {row['sentiment'].title()} (score {row['score']:.3f})")
                        if row['link']:
                            st.markdown(f"[Read article]({row['link']})")
                st.subheader("Sentiment Summary")
                mean_score = news_df['score'].mean()
                st.metric("Average sentiment score", f"{mean_score:.3f}")
                fig_sent = plot_sentiment_counts(news_df['sentiment'])
                st.pyplot(fig_sent)
                plt.close(fig_sent)
            else:
                st.info("No recent news found for this ticker.")

            gc.collect()
            st.success("✅ Analysis completed.")

if __name__ == "__main__":
    main()
