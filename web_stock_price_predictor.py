import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Common stock symbols and names
STOCK_OPTIONS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.",
    "DIS": "The Walt Disney Company",
    "NFLX": "Netflix Inc.",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices Inc.",
    "UBER": "Uber Technologies Inc.",
    "PYPL": "PayPal Holdings Inc."
}

# Page configuration
st.set_page_config(
    page_title="Profit Pulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --background-color: #1a1b23;
        --sidebar-color: #1a1b23;
        --text-color: #ffffff;
        --purple-accent: #7B68EE;
        --card-bg: #2a2b36;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .css-1d391kg {
        background-color: var(--sidebar-color);
    }
    
    h1, h2, h3 {
        color: var(--purple-accent) !important;
    }
    
    .stTextInput input, .stSelectbox select {
        background-color: #2a2b36;
        color: white;
        border: 1px solid var(--purple-accent);
    }
    
    .stDateInput input {
        background-color: #2a2b36;
        color: white;
        border: 1px solid var(--purple-accent);
    }
    
    .dataframe {
        background-color: var(--card-bg);
        color: var(--text-color);
    }
    
    .scrollable-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid var(--purple-accent);
        border-radius: 5px;
        padding: 10px;
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--purple-accent);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--card-bg);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
    }
    
    .stButton button {
        background-color: var(--purple-accent);
        color: white;
    }
    
    .markdown-text-container {
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title('Profit Pulse 📈')
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("📊 Configuration")
    
    selected_symbol = st.selectbox(
        "Select Stock",
        options=list(STOCK_OPTIONS.keys()),
        format_func=lambda x: f"{x} - {STOCK_OPTIONS[x]}",
        help="Select a stock from the list or type to search"
    )
    
    custom_stock = st.text_input(
        "Or Enter Custom Stock Symbol",
        "",
        help="Enter any valid stock symbol (e.g., GOOG, IBM, etc.)"
    )
    
    stock = custom_stock.upper() if custom_stock else selected_symbol
    
    st.subheader("Date Range")
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    selected_start = st.date_input("Start Date", start)
    selected_end = st.date_input("End Date", end)

# Load data and model
try:
    with st.spinner('Fetching stock data...'):
        stock_data = yf.download(stock, selected_start, selected_end)
    
    if stock_data.empty:
        st.error("No data found for the selected stock symbol. Please check the symbol and try again.")
        st.stop()
        
    model = load_model('Latest_Stock_Price_Model.keras')
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Data Overview Section
st.header("📊 Stock Data Overview")

# Safely get and format the latest data
try:
    latest_close = float(stock_data['Close'].iloc[-1])
    previous_close = float(stock_data['Close'].iloc[-2])
    latest_volume = int(stock_data['Volume'].iloc[-1])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${latest_close:.2f}")
    with col2:
        price_change = latest_close - previous_close
        percentage_change = (price_change / previous_close * 100)
        st.metric("Daily Change", f"${price_change:.2f}", f"{percentage_change:.2f}%")
    with col3:
        st.metric("Trading Volume", f"{latest_volume:,}")
except Exception as e:
    st.error(f"Error calculating metrics: {str(e)}")

# Historical Data Display
st.subheader("Historical Data")
with st.container():
    st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    st.dataframe(
        stock_data.sort_index(ascending=False),
        use_container_width=True,
        height=400
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Technical Analysis
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.style.use('dark_background')
    colors = ['#7B68EE', '#9B89F3', '#B8A5F8']
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.rcParams['axes.facecolor'] = '#1a1b23'
    plt.rcParams['figure.facecolor'] = '#1a1b23'
    
    if extra_data:
        plt.plot(full_data.index, full_data['Close'], colors[0], label='Close Price', linewidth=1)
        plt.plot(values.index, values, colors[1], label='100-Day MA', linewidth=1.5)
        plt.plot(extra_dataset.index, extra_dataset, colors[2], label='250-Day MA', linewidth=1.5)
    else:
        plt.plot(full_data.index, full_data['Close'], colors[0], label='Close Price', linewidth=1)
        plt.plot(values.index, values, colors[1], label='Moving Average', linewidth=1.5)
    
    plt.title(f'{stock} Stock Analysis', pad=20, fontsize=12, color='white')
    plt.xlabel('Date', fontsize=10, color='white')
    plt.ylabel('Price ($)', fontsize=10, color='white')
    plt.legend(fontsize=8)
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    return fig

# Calculate and display moving averages
st.header("📈 Technical Analysis")
stock_data['MA_for_250_days'] = stock_data['Close'].rolling(window=250).mean()
stock_data['MA_for_100_days'] = stock_data['Close'].rolling(window=100).mean()

tab1, tab2, tab3 = st.tabs(["250-Day MA", "100-Day MA", "Combined MA"])

with tab1:
    st.pyplot(plot_graph((15,5), stock_data['MA_for_250_days'], stock_data, 0))
with tab2:
    st.pyplot(plot_graph((15,5), stock_data['MA_for_100_days'], stock_data, 0))
with tab3:
    st.pyplot(plot_graph((15,5), stock_data['MA_for_100_days'], stock_data, 1, stock_data['MA_for_250_days']))

# Prediction Section
st.header("🎯 Price Predictions")

# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(stock_data[['Close']])

x_data = []
y_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)
splitting_len = int(len(x_data)*0.7)

x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]
x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

# Generate predictions
with st.spinner('Generating predictions...'):
    try:
        predictions = model.predict(x_test)
        inverse_predictions = scaler.inverse_transform(predictions)
        inverse_y_test = scaler.inverse_transform(y_test)

        # Create prediction dataframe
        plotting_data = pd.DataFrame({
            'Original Values': inverse_y_test.reshape(-1),
            'Predicted Values': inverse_predictions.reshape(-1)
        }, index=stock_data.index[splitting_len+100:])

        # Display predictions
        st.subheader("Prediction Results")
        with st.container():
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
            st.dataframe(
                plotting_data.sort_index(ascending=False),
                use_container_width=True,
                height=400
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Calculate and display accuracy
        mape = np.mean(np.abs((inverse_y_test - inverse_predictions)/inverse_y_test)) * 100
        st.metric("Prediction Accuracy", f"{100-mape:.2f}%")

        # Final visualization
        st.subheader("Complete Price Analysis")
        fig = plt.figure(figsize=(15,5))
        plt.style.use('dark_background')
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.rcParams['axes.facecolor'] = '#1a1b23'
        plt.rcParams['figure.facecolor'] = '#1a1b23'

        colors = ['#7B68EE', '#9B89F3', '#B8A5F8']
        plt.plot(stock_data.index[:splitting_len+100], 
                stock_data['Close'][:splitting_len+100], 
                colors[0], label='Training Data', linewidth=1)
        plt.plot(plotting_data.index, 
                plotting_data['Original Values'], 
                colors[1], label='Original Test Data', linewidth=1)
        plt.plot(plotting_data.index, 
                plotting_data['Predicted Values'], 
                colors[2], label='Predicted Values', linewidth=1)

        plt.title(f'{stock} Stock Price Prediction Analysis', pad=20, fontsize=12, color='white')
        plt.xlabel('Date', fontsize=10, color='white')
        plt.ylabel('Price ($)', fontsize=10, color='white')
        plt.legend(fontsize=8)
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This predictor uses historical data and machine learning to forecast stock prices. 
        Please note that predictions are not guaranteed and should not be used as financial advice.</p>
    </div>
""", unsafe_allow_html=True)