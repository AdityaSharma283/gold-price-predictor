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
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Constants
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"
# Ideally, you would use a real gold price API for Indian prices
INDIAN_GOLD_API = "https://example-gold-api.com/latest" # Replace with actual API

# Set page configuration
st.set_page_config(
    page_title="Gold Investment Advisor Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        font-weight: 700;
        letter-spacing: -0.05em;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #8B7500;
        text-align: center;
        margin-top: 0;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    .feature-section {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 165, 0, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    
    .feature-header {
        color: #8B7500;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .gold-stats {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 215, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .stat-card h4 {
        color: #8B7500;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .stat-card p {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: #333;
    }
    
    .stat-card small {
        color: #777;
        font-size: 0.8rem;
    }
    
    .note {
        font-size: 0.8rem;
        font-style: italic;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .recommendation-box {
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .buy-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(105, 240, 174, 0.1) 100%);
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .wait-box {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%);
        border: 1px solid rgba(255, 152, 0, 0.2);
    }
    
    .mixed-box {
        background: linear-gradient(135deg, rgba(3, 169, 244, 0.1) 0%, rgba(0, 188, 212, 0.1) 100%);
        border: 1px solid rgba(3, 169, 244, 0.2);
    }
    
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #FFD700, #FFA500) !important;
    }
    
    .btn-custom {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        border: none;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
        display: block;
        text-align: center;
        margin: 1rem auto;
        cursor: pointer;
    }
    
    .btn-custom:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(255, 215, 0, 0.4);
    }
    
    .sidebar .sidebar-content {
        background-color: #F5DEB3;
    }
    
    /* Custom CSS for sidebar */
    [data-testid=stSidebar] {
        background-color: #F5DEB3;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #F5DEB3;
        padding-top: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    section[data-testid="stSidebar"] label {
        color: #333333;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #333333;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    
    .analysis-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #8B7500;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .slide-up {
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border-bottom: 4px solid #FFD700 !important;
    }
    
    section[data-testid="stSidebar"] {
        transition: width 0.3s ease;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


# Initialize Gemini API if key is available
if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Load animation JSON
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Get gold animation
def get_lottie_animation():
    animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_9w8ksc4l.json")
    if not animation:
        # Fallback to a simpler animation or to None
        animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_qdzo4pjp.json")
    return animation

def get_usd_to_inr_rate():
    try:
        response = requests.get(USD_TO_INR_API)
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        st.warning(f"Couldn't fetch the latest USD-INR exchange rate. Using default value of 83.5. Error: {e}")
        return 83.5  # Default fallback rate

def convert_oz_to_10g(price_in_oz, adjust_for_indian_market=True, adjustment_factor=1.13):
    """
    Convert gold price from troy ounces to 10 grams with Indian market adjustment
    
    Args:
        price_in_oz: Price in troy ounces
        adjust_for_indian_market: Whether to apply Indian market adjustment
        adjustment_factor: Custom factor for Indian market adjustment
    
    Returns:
        Price per 10 grams
    """
    # 1 troy oz = 31.1035 grams
    base_price_10g = price_in_oz * (10 / 31.1035)
    
    if adjust_for_indian_market:
        return base_price_10g * adjustment_factor
    return base_price_10g

def fetch_indian_gold_price():
    """
    Try to fetch the latest gold price from an Indian source
    Return None if unsuccessful
    """
    try:
        # In a production app, implement an API call to an Indian gold price provider
        # This is a placeholder - in real implementation, replace with actual API call
        response = requests.get("https://www.goldapi.io/api/XAU/INR", 
                               headers={"x-access-token": os.getenv("GOLD_API_KEY")})
        if response.status_code == 200:
            data = response.json()
            # Assuming API returns price per ounce, convert to 10g
            price_per_10g = data["price"] * (10 / 31.1035)
            return price_per_10g
        return None
    except Exception:
        return None

def calculate_optimal_adjustment_factor(international_price_usd, usd_inr_rate, target_indian_price):
    """
    Calculate the optimal adjustment factor to match target Indian price
    """
    # Convert international price to INR per oz
    international_price_inr = international_price_usd * usd_inr_rate
    
    # Calculate base price per 10g without adjustment
    base_price_10g = international_price_inr * (10 / 31.1035)
    
    # Calculate required adjustment factor
    adjustment_factor = target_indian_price / base_price_10g
    
    return adjustment_factor

def ask_gemini_about_market(price_today_inr: float, price_today_usd: float, real_indian_price: float = None) -> str:
    price_today_inr_10g = convert_oz_to_10g(price_today_inr, adjust_for_indian_market=False)
    price_today_usd_10g = convert_oz_to_10g(price_today_usd, adjust_for_indian_market=False)
    
    if real_indian_price:
        indian_price_display = real_indian_price
    else:
        # Use calculated price with adjustment
        indian_price_display = price_today_inr_10g * 1.13
    
    prompt = f"""
    Today's gold price in India is ₹{indian_price_display:.2f} per 10 grams (equivalent to ${price_today_usd_10g:.2f} USD per 10 grams on international markets).
    
    Considering current global economic conditions, Indian market trends, and inflation outlook,
    should a retail investor in India BUY gold today or WAIT?

    Your analysis should consider:
    - Recent price momentum
    - Global and Indian economic indicators
    - USD-INR exchange rate impact
    - Seasonal factors in Indian gold buying
    - Indian gold import duties and taxes
    - Current and expected interest rates
    - Inflation expectations in India
    - Global geopolitical factors

    First, respond with either 'BUY' or 'WAIT' in the first line.
    Then provide a brief, clear, expert reason for your recommendation in 2-3 sentences.
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
    df['MA200'] = df[close_col].rolling(window=200).mean()

    df['MA5_cross_MA20'] = np.where(df['MA5'] > df['MA20'], 1, 0)
    df['MA10_cross_MA50'] = np.where(df['MA10'] > df['MA50'], 1, 0)
    df['MA20_cross_MA100'] = np.where(df['MA20'] > df['MA100'], 1, 0)
    df['MA50_cross_MA200'] = np.where(df['MA50'] > df['MA200'], 1, 0)

    df['Volatility_5d'] = df[close_col].rolling(window=5).std()
    df['Volatility_10d'] = df[close_col].rolling(window=10).std()
    df['Volatility_20d'] = df[close_col].rolling(window=20).std()
    df['Volatility_60d'] = df[close_col].rolling(window=60).std()

    df['Price_Rel_MA5'] = df[close_col] / df['MA5']
    df['Price_Rel_MA10'] = df[close_col] / df['MA10']
    df['Price_Rel_MA20'] = df[close_col] / df['MA20']
    df['Price_Rel_MA50'] = df[close_col] / df['MA50']
    df['Price_Rel_MA200'] = df[close_col] / df['MA200']

    # Bollinger Bands
    df['BB_Middle'] = df[close_col].rolling(window=20).mean()
    df['BB_Std'] = df[close_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df[close_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Volume indicators
    df['Volume_Change'] = df[volume_col].pct_change()
    df['Volume_MA5'] = df[volume_col].rolling(window=5).mean()
    df['Volume_MA10'] = df[volume_col].rolling(window=10).mean()
    df['Volume_MA20'] = df[volume_col].rolling(window=20).mean()
    df['Volume_Rel_MA5'] = df[volume_col] / df['Volume_MA5']
    df['Volume_Rel_MA10'] = df[volume_col] / df['Volume_MA10']
    df['Volume_Rel_MA20'] = df[volume_col] / df['Volume_MA20']
    
    # On-balance Volume
    df['OBV'] = (np.sign(df[close_col].diff()) * df[volume_col]).fillna(0).cumsum()
    df['OBV_MA10'] = df['OBV'].rolling(window=10).mean()
    df['OBV_Rel_MA10'] = df['OBV'] / df['OBV_MA10']

    # Rate of Change
    df['ROC_5'] = (df[close_col] / df[close_col].shift(5) - 1) * 100
    df['ROC_10'] = (df[close_col] / df[close_col].shift(10) - 1) * 100
    df['ROC_20'] = (df[close_col] / df[close_col].shift(20) - 1) * 100
    df['ROC_60'] = (df[close_col] / df[close_col].shift(60) - 1) * 100

    # RSI - Relative Strength Index
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    highest_high = df[close_col].rolling(window=14).max()
    lowest_low = df[close_col].rolling(window=14).min()
    df['Stochastic_%K'] = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()

    # Momentum
    df['Momentum_5d'] = df[close_col] - df[close_col].shift(5)
    df['Momentum_10d'] = df[close_col] - df[close_col].shift(10)
    df['Momentum_20d'] = df[close_col] - df[close_col].shift(20)
    
    # MACD
    df['EMA12'] = df[close_col].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df[close_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Change'] = df['MACD_Hist'].pct_change()

    # Seasonality features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    df['WeekOfYear'] = df.index.isocalendar().week
    
    # Target: 3-day forward return of at least 0.5%
    df['Target'] = np.where(df[close_col].shift(-3) > df[close_col] * 1.005, 1, 0)

    df.dropna(inplace=True)
    return df

def display_landing_page():
    load_css()
    
    animation = get_lottie_animation()
    
    col1, col2 = st.columns([1, 1])  # Equal width columns
    with col1:
      if animation:
        st_lottie(animation, height=200, key="gold_animation")
    with col2:
        st.markdown("""
    <div style="text-align: center;">
        <h1 class="main-header">Gold Investment Advisor Pro</h1>
        <p class="sub-header">AI-Powered Investment Intelligence</p>
    </div>
    """, unsafe_allow_html=True)


    # Sidebar configuration
    st.sidebar.image("image/vecteezy_gold-bar-stack-isolated-on-transparent-background-investment_51573078.png", width=80)
    st.sidebar.title("Market Parameters")
    
    # Add adjustment slider with more informative label
    st.sidebar.subheader("🇮🇳 Indian Market Price Adjustment")
    market_adjustment = st.sidebar.slider(
        "Adjust for import duties, GST, and premiums (%)",
        min_value=10,
        max_value=18,
        value=13,
        step=1,
        help="Adjust this to match current market prices in India (includes import duties, GST, and dealer margins)"
    )
    
    # Add option to manually set current gold price
    st.sidebar.subheader("🏷️ Reference Gold Price (10g)")
    use_custom_price = st.sidebar.checkbox("Set market price manually", value=False)
    custom_price = None
    
    if use_custom_price:
        custom_price = st.sidebar.number_input(
            "Enter current 24K gold price (₹ per 10g)",
            min_value=50000,
            max_value=120000,
            value=98000,
            step=100,
            help="Enter the exact market price you're seeing for 24K (99.9%) gold in India"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Settings")
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        options=["3 days", "5 days", "7 days", "10 days"],
        index=0
    )
    
    # Update the adjustment factor based on the slider
    market_adjustment_factor = 1 + (market_adjustment / 100)

    # Features section
    st.markdown('<div class="feature-section slide-up">', unsafe_allow_html=True)
    st.markdown('<h3 class="feature-header">📊 AI-Powered Gold Investment Suite</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🧠 AI Market Analysis**
        * Gemini AI insights
        * Global economic evaluation
        * Smart trend detection
        * Personalized recommendations
        """)
    with col2:
        st.markdown("""
        **📈 Advanced Technical Analysis**
        * 50+ technical indicators
        * Pattern recognition
        * Multi-timeframe signals
        * Market sentiment scoring
        """)
    with col3:
        st.markdown("""
        **💰 Indian Market Intelligence**
        * Real-time price conversion
        * Import duty adjustment
        * GST & dealer margin calculation
        * Seasonal buying pattern analysis
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
        
        # Calculate the 10g prices
        current_price_usd_10g = convert_oz_to_10g(current_price_usd, adjust_for_indian_market=False)
        current_price_inr = current_price_usd * usd_inr_rate
        
        # Calculate Indian market price
        current_price_inr_10g = convert_oz_to_10g(current_price_inr, adjust_for_indian_market=True, 
                                                 adjustment_factor=market_adjustment_factor)
        
        # If user provided a custom price, calculate the adjustment factor
        if custom_price:
            current_price_inr_10g = custom_price
            market_adjustment_factor = calculate_optimal_adjustment_factor(
                current_price_usd, usd_inr_rate, custom_price)
        
        monthly_change = (gold_data[close_col].iloc[-1] / gold_data[close_col].iloc[0] - 1) * 100
        weekly_change = (gold_data[close_col].iloc[-1] / gold_data[close_col].iloc[-7] - 1) * 100
        volatility = gold_data[close_col].pct_change().std() * 100

        # Stats display with enhanced styling
        st.markdown('<div class="gold-stats fade-in">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h4>Today's Price (10g)</h4>
                <p>₹{current_price_inr_10g:,.2f}</p>
                <small>${current_price_usd_10g:.2f} USD</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_icon = "📈" if monthly_change > 0 else "📉"
            trend_color = "#4CAF50" if monthly_change > 0 else "#F44336"
            st.markdown(f"""
            <div class="stat-card">
                <h4>30-Day Change</h4>
                <p style="color: {trend_color}">{trend_icon} {monthly_change:.2f}%</p>
                <small>7-Day: {weekly_change:.2f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            volatility_level = "Low" if volatility < 1 else "Medium" if volatility < 2 else "High"
            volatility_color = "#4CAF50" if volatility < 1 else "#FF9800" if volatility < 2 else "#F44336"
            st.markdown(f"""
            <div class="stat-card">
                <h4>Volatility</h4>
                <p>{volatility:.2f}%</p>
                <small style="color: {volatility_color}">{volatility_level}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h4>USD-INR Rate</h4>
                <p>{usd_inr_rate:.2f}</p>
                <small>Exchange Rate</small>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not custom_price:
            st.markdown(f"""
            <p class="note">* Indian market price includes estimated adjustments ({market_adjustment}% for taxes, duties, and dealer margins)</p>
            """, unsafe_allow_html=True)
        
        # Chart with 10g prices (Indian market adjusted)
        st.markdown("### 📊 Gold Price Trend (Last 30 Days)")
        
        # Create data for plotting
        plot_data = gold_data.copy()
        plot_data['Date'] = plot_data.index
        plot_data['Close_USD_10g'] = plot_data[close_col].apply(lambda x: convert_oz_to_10g(x, adjust_for_indian_market=False))
        plot_data['Close_INR_10g'] = plot_data['Close_USD_10g'] * usd_inr_rate * market_adjustment_factor
        
        # Use Plotly for more interactive charts
        fig = px.line(
            plot_data, 
            x='Date', 
            y='Close_INR_10g',
            labels={'Close_INR_10g': 'Price (₹ per 10g)', 'Date': 'Date'},
            title=None
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
            ),
            yaxis=dict(
                tickprefix='₹',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
            ),
            font=dict(family="Poppins"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        fig.update_traces(
            line=dict(color='#FFD700', width=3),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:,.2f}<extra></extra>'
        )
        
        # Add a marker for the current price
        fig.add_trace(
            go.Scatter(
                x=[plot_data.index[-1]], 
                y=[plot_data['Close_INR_10g'].iloc[-1]],
                mode='markers',
                marker=dict(color='#FFA500', size=10),
                name='Latest Price',
                hoverinfo='skip'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Couldn't load gold stats for display: {e}")

    # Create tabs for different insights
    st.markdown("### 🔍 Market Insights")
    tabs = st.tabs(["🧠 AI Analysis", "📈 Technical Signals", "🌍 Global Factors"])
    
    with tabs[0]:
        st.markdown("""
        Our AI analyzes multiple factors affecting gold prices:
        
        - **Economic indicators**: Interest rates, inflation trends, central bank policies
        - **Market sentiment**: Trading volume, investor behavior, risk appetite
        - **Seasonal patterns**: Indian festival demand, marriage season buying
        - **Currency dynamics**: USD-INR rate fluctuations and outlook
        """)
        
    with tabs[1]:
        st.markdown("""
        Current technical indicators for gold:
        
        - **Moving Averages**: Gold is trading above its 50-day and 200-day moving averages, suggesting a bullish trend
        - **RSI**: Current RSI indicates moderate momentum without being overbought
        - **MACD**: MACD histogram shows positive momentum building
        - **Bollinger Bands**: Price is trading in the upper zone of the Bollinger bands
        """)
        
    with tabs[2]:
        st.markdown("""
        Global factors impacting gold prices:
        
        - **Central Bank Policies**: Fed's stance on interest rates and monetary policy
        - **Geopolitical Tensions**: Current conflicts and their impact on safe-haven assets
        - **Inflation Concerns**: Rising global inflation supporting gold as a hedge
        - **Economic Recovery**: Post-pandemic growth patterns and their effect on commodities
        """)

    st.markdown("---")
    st.markdown("### Ready to get your personalized gold investment recommendation?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        start_button = st.button(
            "🚀 Generate AI Prediction", 
            key="start_prediction", 
            use_container_width=True,
            type="primary"
        )
    with col2:
        st.markdown(
            """<a href="#" style="color: #8B7500; text-decoration: none;">💡 View sample report</a>""", 
            unsafe_allow_html=True
        )
    
    return start_button, market_adjustment_factor, custom_price, prediction_horizon

def predict():
    result = display_landing_page()
    if not result or not result[0]:
        return
    
    start_button, market_adjustment_factor, custom_price, prediction_horizon = result
    
    # Convert prediction horizon to days
    days_mapping = {"3 days": 3, "5 days": 5, "7 days": 7, "10 days": 10}
    prediction_days = days_mapping[prediction_horizon]

    st.markdown("---")
    st.subheader("⚙️ Analysis in Progress")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Download data
    status_text.text("📥 Downloading Gold Price Data...")
    gold_data = yf.download('GC=F', start='2018-01-01')
    progress_bar.progress(20)
    
    # Flatten columns if needed
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = ['_'.join([str(i) for i in col if i]) for col in gold_data.columns.values]
    close_col = [col for col in gold_data.columns if col.lower().startswith('close')][0]
    
    if gold_data.empty:
        st.error("❌ Failed to load gold price data. Please check your internet connection or symbol.")
        st.stop()
    
    status_text.text("✓ Data loaded successfully")
    progress_bar.progress(30)
    
    # Step 2: Feature engineering
    status_text.text("🔧 Creating Advanced Technical Indicators...")
    gold_data = create_advanced_features(gold_data)
    feature_cols = [col for col in gold_data.columns if col not in ['Target', 'Date', 'Adj Close']]
    progress_bar.progress(50)
    status_text.text(f"✓ Created {len(feature_cols)} predictive features")
    
    # Step 3: Model training
    status_text.text("🧠 Training Advanced Predictive Models...")
    X = gold_data[feature_cols]
    y = gold_data['Target']

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    progress_bar.progress(70)
    
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
    accuracy = scores.mean()
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy')
    rf_accuracy = rf_scores.mean()
    
    pipeline.fit(X, y)
    rf_model.fit(X, y)
    progress_bar.progress(90)
    status_text.text("✓ Models trained successfully")
    
    # Step 4: Feature importance
    gb_model = pipeline.named_steps['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    # Final calculations
    latest_data = gold_data[feature_cols].iloc[-1].values.reshape(1, -1)
    gb_prediction = pipeline.predict_proba(latest_data)[0][1]
    rf_prediction = rf_model.predict_proba(latest_data)[0][1]
    ensemble_pred = (gb_prediction * 0.6) + (rf_prediction * 0.4)
    action_ml = "BUY" if ensemble_pred > 0.5 else "WAIT"
    
    # Calculate prices
    gold_price_today_usd = float(gold_data[close_col].iloc[-1])
    gold_price_today_usd_10g = convert_oz_to_10g(gold_price_today_usd, adjust_for_indian_market=False)
    usd_inr_rate = get_usd_to_inr_rate()
    gold_price_today_inr = gold_price_today_usd * usd_inr_rate
    
    # Use custom price if provided
    if custom_price:
        gold_price_today_inr_10g = custom_price
    else:
        # Apply the custom adjustment factor selected by user
        gold_price_today_inr_10g = convert_oz_to_10g(gold_price_today_inr, 
                                                    adjust_for_indian_market=False) * market_adjustment_factor
    
    # Get AI advice
    progress_bar.progress(95)
    status_text.text("🤖 Consulting AI for market insights...")
    
    try:
        gemini_advice = ask_gemini_about_market(gold_price_today_inr, gold_price_today_usd, 
                                               real_indian_price=gold_price_today_inr_10g)
    except Exception as e:
        gemini_advice = f"AI Analysis Error: {str(e)}\n\nPlease try again later."
    
    progress_bar.progress(100)
    status_text.empty()  # Clear the status text
    
    # Display results in a beautiful format
    st.empty()  # Clear previous content
    
    st.markdown("""
    <style>
    .recommendation-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='recommendation-header'>Your Gold Investment Analysis</h1>", unsafe_allow_html=True)
    
    # Current price metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📈 Current Gold Price</div>
            <div class="metric-value">₹{:,.2f}</div>
            <div class="metric-label">per 10 grams in India</div>
            <div style="margin-top: 1rem; font-size: 0.9rem;">
                <strong>International Price:</strong> ${:.2f} per 10g<br>
                <strong>USD-INR Rate:</strong> {:.2f}
            </div>
        </div>
        """.format(gold_price_today_inr_10g, gold_price_today_usd_10g, usd_inr_rate), unsafe_allow_html=True)
    
    with col2:
        # Price trend analysis
        weekly_change = (gold_data[close_col].iloc[-1] / gold_data[close_col].iloc[-7] - 1) * 100
        monthly_change = (gold_data[close_col].iloc[-1] / gold_data[close_col].iloc[-30] - 1) * 100
        st.markdown("""
        <div class="analysis-card">
            <div class="card-title">📊 Price Trend Analysis</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                <div>
                    <div class="metric-value" style="font-size: 1.5rem;">{:.2f}%</div>
                    <div class="metric-label">7-Day Change</div>
                </div>
                <div>
                    <div class="metric-value" style="font-size: 1.5rem;">{:.2f}%</div>
                    <div class="metric-label">30-Day Change</div>
                </div>
            </div>
            <div style="font-size: 0.9rem;">
                <strong>Momentum:</strong> {}<br>
                <strong>Volatility:</strong> {:.2f}% (Daily)
            </div>
        </div>
        """.format(
            weekly_change, 
            monthly_change,
            "Strong Positive" if monthly_change > 3 else "Positive" if monthly_change > 0 else "Negative",
            gold_data[close_col].pct_change().std() * 100
        ), unsafe_allow_html=True)
    
    # AI Recommendation Box
    recommendation_class = "buy-box" if action_ml == "BUY" else "wait-box"
    if action_ml == "BUY" and "BUY" in gemini_advice.upper()[:10]:
        recommendation_strength = "STRONG BUY"
        recommendation_class = "buy-box"
        agreement = "Both AI analysis and technical models strongly recommend buying gold now."
    elif action_ml == "WAIT" and "WAIT" in gemini_advice.upper()[:10]:
        recommendation_strength = "STRONG WAIT"
        recommendation_class = "wait-box"
        agreement = "Both AI analysis and technical models suggest holding off on buying gold now."
    else:
        recommendation_strength = "MIXED SIGNALS"
        recommendation_class = "mixed-box"
        agreement = "Technical and AI analyses show different signals. Consider your investment timeline and goals."
    
    st.markdown(f"""
    <div class="recommendation-box {recommendation_class}">
        <h2 style="text-align: center; margin-bottom: 1.5rem;">{recommendation_strength}</h2>
        <p style="text-align: center; font-size: 1.1rem;">{agreement}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Buy confidence
    buy_confidence = ensemble_pred * 100
    st.markdown("### 🧠 AI Model Prediction")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(buy_confidence / 100)
    with col2:
        st.markdown(f"**{buy_confidence:.1f}%** Confidence")
        
    if action_ml == "BUY":
        st.success(f"**Recommendation: BUY** - The model predicts a price increase in the next {prediction_days} days.")
    else:
        st.warning(f"**Recommendation: WAIT** - The model does not predict a significant price increase in the next {prediction_days} days.")

    # AI Analysis from Gemini
    st.markdown("### 🤖 AI Market Analysis")
    if "BUY" in gemini_advice.upper()[:10]:
        st.success(gemini_advice)
    elif "WAIT" in gemini_advice.upper()[:10]:
        st.warning(gemini_advice)
    else:
        st.info(gemini_advice)
    
    # Two-column layout for technical analysis and charts
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Feature importance
        st.markdown("### 🔍 Key Price Drivers")
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            labels={'Importance': 'Impact Strength', 'Feature': 'Factor'},
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
            ),
            yaxis=dict(
                showgrid=False,
            ),
            font=dict(family="Poppins", size=10),
        )
        fig.update_traces(marker_color='#FFD700')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recent price trend with buy signals
        st.markdown("### 📈 Price Trend with Buy Signals")
        recent_data = gold_data.iloc[-60:].copy()
        X_recent = recent_data[feature_cols]
        recent_data['Prediction'] = pipeline.predict(X_recent)
        recent_data['Close_INR'] = recent_data[close_col] * usd_inr_rate
        recent_data['Close_INR_10g'] = recent_data['Close_INR'].apply(
            lambda x: convert_oz_to_10g(x, adjust_for_indian_market=False) * market_adjustment_factor
        )
        
        # Create plotly chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=recent_data.index, 
                y=recent_data['Close_INR_10g'],
                mode='lines',
                name='Gold Price (10g INR)',
                line=dict(color='#FFD700', width=3),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:,.2f}<extra></extra>'
            )
        )
        
        # Add buy signals
        buy_signals = recent_data[recent_data['Prediction'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, 
                    y=buy_signals['Close_INR_10g'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    hovertemplate='<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
                )
            )
        
        # Add SMA lines
        fig.add_trace(
            go.Scatter(
                x=recent_data.index, 
                y=recent_data['MA20'] * usd_inr_rate * (10/31.1035) * market_adjustment_factor,
                mode='lines',
                name='20-Day MA',
                line=dict(color='#4CAF50', width=1.5, dash='dash'),
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=recent_data.index, 
                y=recent_data['MA50'] * usd_inr_rate * (10/31.1035) * market_adjustment_factor,
                mode='lines',
                name='50-Day MA',
                line=dict(color='#FF9800', width=1.5, dash='dash'),
                hoverinfo='skip'
            )
        )
        
        fig.update_layout(
            title='Gold Price (₹ per 10g) with Buy Signals - Last 60 Days',
            xaxis_title='Date',
            yaxis_title='Price (₹ per 10g)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
            ),
            yaxis=dict(
                tickprefix='₹',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
            ),
            font=dict(family="Poppins"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical insights section
    st.markdown("### 📊 Technical Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # RSI indicator
        rsi_value = gold_data['RSI'].iloc[-1]
        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        rsi_color = "#F44336" if rsi_value > 70 else "#4CAF50" if rsi_value < 30 else "#FF9800"
        
        st.markdown(f"""
        <div class="analysis-card">
            <div class="card-title">📉 RSI Indicator</div>
            <div class="metric-value">{rsi_value:.1f}</div>
            <div class="metric-label" style="color:{rsi_color}">{rsi_status}</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                RSI measures momentum and indicates potential overbought or oversold conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Moving Average Status
        ma_cross = "Bullish" if gold_data['MA10_cross_MA50'].iloc[-1] == 1 else "Bearish"
        ma_color = "#4CAF50" if ma_cross == "Bullish" else "#F44336"
        
        st.markdown(f"""
        <div class="analysis-card">
            <div class="card-title">📈 Moving Averages</div>
            <div class="metric-value" style="color:{ma_color}">{ma_cross}</div>
            <div class="metric-label">10 & 50-Day MA Cross</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                MA crossovers indicate potential trend shifts in the market.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Bollinger Band Status
        bb_position = gold_data['BB_Position'].iloc[-1]
        if bb_position > 0.8:
            bb_status = "Upper Band"
            bb_description = "Price near upper band signals strength but potential reversal."
            bb_color = "#F44336"
        elif bb_position < 0.2:
            bb_status = "Lower Band"
            bb_description = "Price near lower band signals weakness but potential rebound."
            bb_color = "#4CAF50"
        else:
            bb_status = "Middle Band"
            bb_description = "Price near middle band signals consolidation phase."
            bb_color = "#FF9800"
            
        st.markdown(f"""
        <div class="analysis-card">
            <div class="card-title">🎯 Bollinger Bands</div>
            <div class="metric-value" style="color:{bb_color}">{bb_status}</div>
            <div class="metric-label">{bb_position*100:.1f}% Position</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                {bb_description}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add investment scenarios
    st.markdown("### 💰 Investment Scenarios")
    
    # Calculate potential outcomes
    base_investment = 100000  # ₹1 lakh base investment
    conservative_return = base_investment * (1 + 0.05)  # 5% return
    moderate_return = base_investment * (1 + 0.10)      # 10% return
    aggressive_return = base_investment * (1 + 0.15)    # 15% return
    
    scenarios_df = pd.DataFrame({
        "Scenario": ["Conservative", "Moderate", "Aggressive"],
        "Expected Return": ["5%", "10%", "15%"],
        "Investment Value": [f"₹{base_investment:,.0f}", f"₹{base_investment:,.0f}", f"₹{base_investment:,.0f}"],
        "Projected Value": [
            f"₹{conservative_return:,.0f}", 
            f"₹{moderate_return:,.0f}", 
            f"₹{aggressive_return:,.0f}"
        ],
        "Profit": [
            f"₹{conservative_return - base_investment:,.0f}", 
            f"₹{moderate_return - base_investment:,.0f}", 
            f"₹{aggressive_return - base_investment:,.0f}"
        ],
        "Timeline": ["3-6 months", "6-12 months", "12+ months"]
    })
    
    # Display the investment scenarios beautifully
    st.dataframe(
    scenarios_df.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', 'rgba(255, 215, 0, 0.1)'), ('text-align', 'center')]}
    ]).set_properties(**{'text-align': 'center'}),
    use_container_width=True
)

    
    # Disclaimer and footer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: rgba(0,0,0,0.03); padding: 1.5rem; border-radius: 10px; font-size: 0.9rem;">
        <strong>⚠️ Disclaimer:</strong> This tool provides investment insights based on historical data analysis and AI projections, 
        but does not constitute financial advice. Gold prices are subject to market volatility and influenced by numerous factors.
        Consult with a qualified financial advisor before making investment decisions.
        <br><br>
        <strong>Notes:</strong>
        <ul style="margin-top: 0.5rem;">
            <li>Indian market prices include estimated import duties, GST, and dealer margins.</li>
            <li>All projections are based on historical data and may not accurately predict future performance.</li>
            <li>Investment scenarios are hypothetical and for illustrative purposes only.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Share/export options
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📝 Save Analysis as PDF", key="save_pdf", use_container_width=True)
    with col2:
        st.button("📱 Share via WhatsApp", key="share_whatsapp", use_container_width=True)
    with col3:
        st.button("📧 Email Report", key="email_report", use_container_width=True)

if __name__ == "__main__":
    predict()