import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"

if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

def get_usd_to_inr_rate():
    try:
        response = requests.get(USD_TO_INR_API)
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        st.warning(f"Couldn't fetch the latest USD-INR exchange rate. Using default value of 83.5. Error: {e}")
        return 83.5  # Default fallback rate

def ask_gemini_about_market(price_today_inr: float, price_today_usd: float) -> str:
    prompt = f"""
    Today's gold price is {price_today_usd:.2f} USD (₹{price_today_inr:.2f} INR).
    Considering current global economic conditions, gold market trends, and inflation outlook,
    should a retail investor in India BUY gold today or WAIT?

    Your analysis should consider:
    - Recent price momentum
    - Global economic indicators
    - USD-INR exchange rate impact
    - Seasonal factors

    First, respond with either 'BUY' or 'WAIT' in the first line.
    Then provide a brief, clear reason for your recommendation in 1-2 sentences.
    """
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def create_advanced_features(df):
    df = df.copy()

    # --- Flatten MultiIndex columns if needed ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]

    # --- Try to find the correct 'Close' and 'Volume' column ---
    close_col = None
    volume_col = None
    for col in df.columns:
        if col.lower().startswith('close'):
            close_col = col
        if col.lower().startswith('volume'):
            volume_col = col
    if close_col is None or volume_col is None:
        raise ValueError("Could not find 'Close' or 'Volume' column after flattening.")

    df['Price_Change'] = df[close_col].pct_change()
    df['Price_Change_3d'] = df[close_col].pct_change(periods=3)
    df['Price_Change_5d'] = df[close_col].pct_change(periods=5)
    df['Price_Change_10d'] = df[close_col].pct_change(periods=10)

    df['MA5'] = df[close_col].rolling(window=5).mean()
    df['MA10'] = df[close_col].rolling(window=10).mean()
    df['MA20'] = df[close_col].rolling(window=20).mean()
    df['MA50'] = df[close_col].rolling(window=50).mean()
    df['MA100'] = df[close_col].rolling(window=100).mean()

    df['MA5_cross_MA20'] = np.where(df['MA5'] > df['MA20'], 1, 0)
    df['MA10_cross_MA50'] = np.where(df['MA10'] > df['MA50'], 1, 0)
    df['MA20_cross_MA100'] = np.where(df['MA20'] > df['MA100'], 1, 0)

    df['Volatility_5d'] = df[close_col].rolling(window=5).std()
    df['Volatility_10d'] = df[close_col].rolling(window=10).std()
    df['Volatility_20d'] = df[close_col].rolling(window=20).std()

    df['Price_Rel_MA5'] = df[close_col] / df['MA5']
    df['Price_Rel_MA10'] = df[close_col] / df['MA10']
    df['Price_Rel_MA20'] = df[close_col] / df['MA20']
    df['Price_Rel_MA50'] = df[close_col] / df['MA50']

    df['BB_Middle'] = df[close_col].rolling(window=20).mean()
    df['BB_Std'] = df[close_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df[close_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    df['Volume_Change'] = df[volume_col].pct_change()
    df['Volume_MA5'] = df[volume_col].rolling(window=5).mean()
    df['Volume_MA10'] = df[volume_col].rolling(window=10).mean()
    df['Volume_Rel_MA5'] = df[volume_col] / df['Volume_MA5']

    df['ROC_5'] = (df[close_col] / df[close_col].shift(5) - 1) * 100
    df['ROC_10'] = (df[close_col] / df[close_col].shift(10) - 1) * 100
    df['ROC_20'] = (df[close_col] / df[close_col].shift(20) - 1) * 100

    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Momentum_5d'] = df[close_col] - df[close_col].shift(5)
    df['Momentum_10d'] = df[close_col] - df[close_col].shift(10)
    df['Momentum_20d'] = df[close_col] - df[close_col].shift(20)

    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

    df['Target'] = np.where(df[close_col].shift(-3) > df[close_col] * 1.005, 1, 0)

    df.dropna(inplace=True)
    return df

def display_landing_page():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #E5C100;
        text-align: center;
        margin-top: 0;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .feature-section {
        background-color: rgba(255, 215, 0, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .feature-header {
        color: #FFD700;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    .gold-stats {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    .stat-card {
        background-color: rgba(0,0,0,0.03);
        padding: 1rem;
        border-radius: 8px;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
    <h1 class="main-header">🏆 Gold Investment Advisor</h1>
    <p class="sub-header">Powered by Advanced ML & Gemini AI</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="feature-header">📊 Advanced Decision Support</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **AI Analysis**
        * Gemini AI market insights
        * Economic trend evaluation
        * Smart recommendations
        """)
    with col2:
        st.markdown("""
        **Technical Analysis**
        * 25+ technical indicators
        * Historic pattern matching
        * Multi-timeframe signals
        """)
    with col3:
        st.markdown("""
        **Price Analytics**
        * USD to INR conversion
        * 3-day price forecasting
        * Volatility assessment
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        gold_data = yf.download('GC=F', period='1mo')
        # Flatten columns if needed
        if isinstance(gold_data.columns, pd.MultiIndex):
            gold_data.columns = ['_'.join([str(i) for i in col if i]) for col in gold_data.columns.values]
        close_col = [col for col in gold_data.columns if col.lower().startswith('close')][0]
        usd_inr_rate = get_usd_to_inr_rate()
        current_price_usd = gold_data[close_col].iloc[-1]
        current_price_inr = current_price_usd * usd_inr_rate
        current_price_inr_10g = (current_price_inr / 31.1035) * 10
        monthly_change = (gold_data[close_col].iloc[-1] / gold_data[close_col].iloc[0] - 1) * 100
        volatility = gold_data[close_col].pct_change().std() * 100

        st.markdown('<div class="gold-stats">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <h4>Current Price</h4>
            <p>₹{current_price_inr:.2f} INR<br>${current_price_usd:.2f} USD</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <h4>30-Day Change</h4>
            <p>{monthly_change:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <h4>Volatility</h4>
            <p>{volatility:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <h4>USD-INR Rate</h4>
            <p>{usd_inr_rate:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gold_data.index, gold_data[close_col], color='#FFD700')
        ax.set_title('Gold Price Trend (Last 30 Days)', color='#333333')
        ax.set_ylabel('Price (USD)', color='#333333')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Couldn't load gold stats for display: {e}")

    st.markdown("---")
    st.markdown("### Ready to get your gold investment recommendation?")
    start_button = st.button(" Generate Prediction", key="start_prediction", use_container_width=True)
    return start_button

def predict():
    if not display_landing_page():
        return

    st.markdown("---")
    st.subheader("⚙️ Analysis in Progress")
    with st.spinner("📥 Downloading Gold Price Data..."):
        gold_data = yf.download('GC=F', start='2018-01-01')
        # Flatten columns if needed
        if isinstance(gold_data.columns, pd.MultiIndex):
            gold_data.columns = ['_'.join([str(i) for i in col if i]) for col in gold_data.columns.values]
        close_col = [col for col in gold_data.columns if col.lower().startswith('close')][0]
        if gold_data.empty:
            st.error(" Failed to load gold price data. Please check your internet connection or symbol.")
            st.stop()
        st.success(f"✅ Data loaded: {len(gold_data)} records spanning {(gold_data.index[-1] - gold_data.index[0]).days} days")

    with st.spinner("🔧 Creating Advanced Technical Indicators..."):
        gold_data = create_advanced_features(gold_data)
        feature_cols = [col for col in gold_data.columns if col not in ['Target', 'Date', 'Adj Close']]
        st.success(f"✅ Created {len(feature_cols)} predictive features")

    with st.spinner(" Training Advanced Predictive Models..."):
        X = gold_data[feature_cols]
        y = gold_data['Target']

        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        tscv = TimeSeriesSplit(n_splits=5)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ))
        ])
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        accuracy = scores.mean()
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy')
        rf_accuracy = rf_scores.mean()
        pipeline.fit(X, y)
        rf_model.fit(X, y)
        st.success(f" Model Cross-Validation Accuracy: **{accuracy*100:.2f}%**")
        st.success(f" Secondary Model Accuracy: **{rf_accuracy*100:.2f}%**")
        gb_model = pipeline.named_steps['model']
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': gb_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)

    latest_data = gold_data[feature_cols].iloc[-1].values.reshape(1, -1)
    gb_prediction = pipeline.predict_proba(latest_data)[0][1]
    rf_prediction = rf_model.predict_proba(latest_data)[0][1]
    ensemble_pred = (gb_prediction * 0.6) + (rf_prediction * 0.4)
    action_ml = "BUY" if ensemble_pred > 0.5 else "WAIT"

    gold_price_today_usd = float(gold_data[close_col].iloc[-1])
    usd_inr_rate = get_usd_to_inr_rate()
    gold_price_today_inr = gold_price_today_usd * usd_inr_rate

    with st.spinner(" Consulting AI..."):
        try:
            gemini_advice = ask_gemini_about_market(gold_price_today_inr, gold_price_today_usd)
        except Exception as e:
            gemini_advice = f"Gemini API Error: {str(e)}"

    st.markdown("---")
    st.subheader("📊 Today's Gold Investment Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📈 Gold Price (INR)", value=f"₹{gold_price_today_inr:.2f}")
    with col2:
        st.metric(label="📈 Gold Price (USD)", value=f"${gold_price_today_usd:.2f}")

    buy_confidence = ensemble_pred * 100
    st.markdown("###  AI Model Prediction")
    if action_ml == "BUY":
        st.progress(buy_confidence / 100)
        st.success(f"**Recommendation: BUY** with {buy_confidence:.1f}% confidence")
    else:
        st.progress(1 - (buy_confidence / 100))
        st.warning(f"**Recommendation: WAIT** with {100-buy_confidence:.1f}% confidence")

    st.markdown("###  AI Analysis")
    if "BUY" in gemini_advice.upper()[:10]:
        st.success(gemini_advice)
    elif "WAIT" in gemini_advice.upper()[:10]:
        st.warning(gemini_advice)
    else:
        st.info(gemini_advice)

    st.markdown("### 📈 Recent Price Trend")
    recent_data = gold_data.iloc[-30:].copy()
    X_recent = recent_data[feature_cols]
    recent_data['Prediction'] = pipeline.predict(X_recent)
    recent_data['Close_INR'] = recent_data[close_col] * usd_inr_rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recent_data.index, recent_data['Close_INR'], label='Gold Price (INR)', color='#FFD700', linewidth=2)
    buy_signals = recent_data[recent_data['Prediction'] == 1]
    if not buy_signals.empty:
        ax.scatter(buy_signals.index, buy_signals['Close_INR'], color='green', label='Buy Signal', marker='^', s=100)
    ax.set_title('Gold Price (INR) with Buy Signals - Last 30 Days')
    ax.set_ylabel('Price (INR)')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔍 Final Recommendation")
    if action_ml == "BUY" and "BUY" in gemini_advice.upper()[:10]:
        st.success("**STRONG BUY** - Both technical analysis and AI suggest favorable buying conditions.")
    elif action_ml == "WAIT" and "WAIT" in gemini_advice.upper()[:10]:
        st.warning("**STRONG WAIT** - Both technical analysis and AI suggest holding off on buying gold now.")
    else:
        st.info("**MIXED SIGNALS** - Consider consulting with a financial advisor before making investment decisions.")

    st.caption("This tool provides investment insights based on historical data and AI analysis, but does not constitute financial advice.")
   

if __name__ == "__main__":
    predict()
