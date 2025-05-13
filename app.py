import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

from prediction_model import StockPredictor
from sentiment_analyzer import SentimentAnalyzer
from data_manager import DataManager
from utils import get_stock_symbol, format_price

# Page configuration
st.set_page_config(
    page_title="Stock Sense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import time

# Show splash screen
with st.spinner("Loading Stock Sense..."):
    time.sleep(2.5)  # Show splash screen for 2.5 seconds

# Initialize components
data_manager = DataManager()
predictor = StockPredictor(data_manager)
sentiment_analyzer = SentimentAnalyzer()

# App title and description
st.title("Stock Sense")
st.markdown("Created With Love By Aarav")

# Search bar for stock tickers
st.subheader("Search for a stock")

search_query = st.text_input(
    "Enter an Indian stock ticker (e.g., RELIANCE.NS, TCS.NS, INFY.NS)",
    placeholder="Type stock symbol",
    key="stock_search",
    max_chars=20
)
search_button = st.button("Search", type="primary")

# Process the search
if search_query and search_button:
    # Convert input to proper Yahoo Finance symbol format
    stock_symbol = get_stock_symbol(search_query)
    
    try:
        # Get stock data
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Check if valid stock
        if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
            st.error(f"Could not find valid data for ticker: {stock_symbol}. Please check the symbol and try again.")
        else:
            # Display current stock information
            st.header(f"{info.get('shortName', stock_symbol)}")
            
            # Add time period selection
            time_periods = {
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365,
                "2 Years": 730,
                "3 Years": 1095
            }
            
            selected_period = st.selectbox(
                "Select historical time period:",
                options=list(time_periods.keys()),
                index=2  # Default to 1 Year
            )
            
            days_to_fetch = time_periods[selected_period]
            
            col1, col2 = st.columns(2)
            
            # Current price and basic info
            with col1:
                current_price = info['regularMarketPrice']
                previous_close = info.get('previousClose', 0)
                price_change = current_price - previous_close
                price_change_percent = (price_change / previous_close) * 100 if previous_close != 0 else 0
                
                change_color = "green" if price_change >= 0 else "red"
                change_icon = "↗" if price_change >= 0 else "↘"
                
                st.metric(
                    label="Current Price", 
                    value=f"₹{format_price(current_price)}",
                    delta=f"{change_icon} ₹{abs(price_change):.2f} ({abs(price_change_percent):.2f}%)",
                    delta_color="normal" if price_change >= 0 else "inverse"
                )
                
                st.caption(f"Day Range: ₹{format_price(info.get('dayLow', 0))} - ₹{format_price(info.get('dayHigh', 0))}")
                st.caption(f"52 Week Range: ₹{format_price(info.get('fiftyTwoWeekLow', 0))} - ₹{format_price(info.get('fiftyTwoWeekHigh', 0))}")
            
            # Get historical data for the selected time period (using timezone-aware dates in UTC)
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days_to_fetch)
            historical_data = stock.history(start=start_date, end=end_date)
            
            # Ensure the datetime index is timezone-aware in UTC for consistency
            if not historical_data.empty and len(historical_data) > 0:
                if hasattr(historical_data.index[0], 'tzinfo') and historical_data.index[0].tzinfo is not None:
                    if historical_data.index[0].tzinfo != pytz.UTC:
                        historical_data.index = historical_data.index.tz_convert(pytz.UTC)
                else:
                    historical_data.index = historical_data.index.tz_localize(pytz.UTC)
            
            if not historical_data.empty:
                # Prepare data for prediction
                data_manager.update_stock_data(stock_symbol, historical_data)
                
                # Get sentiment analysis
                news_sentiment = sentiment_analyzer.analyze_stock_sentiment(stock_symbol, info.get('shortName', ''))
                
                # Make prediction for each of the next 7 days
                prediction_days = 7  # Always predict for next 7 days
                try:
                    prediction_results, confidence = predictor.predict_stock_price(
                        stock_symbol, 
                        prediction_days, 
                        sentiment_score=news_sentiment['compound_score']
                    )
                    st.write(f"Debug: Prediction successful for {stock_symbol}")
                    
                    # Extract the 7-day prediction (final prediction)
                    final_date, final_predicted_price = prediction_results[-1]
                except Exception as e:
                    st.error(f"Prediction Error Detail: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    # Create fallback prediction data (current price for all days)
                    today = datetime.now(pytz.UTC)
                    prediction_results = [(today + timedelta(days=i+1), current_price) for i in range(7)]
                    final_predicted_price = current_price
                    confidence = 0
                
                # Display final prediction in the metric
                with col2:
                    st.metric(
                        label=f"Predicted Price (7 days ahead)",
                        value=f"₹{format_price(final_predicted_price)}",
                        delta=f"{'+' if final_predicted_price > current_price else '-'}₹{abs(final_predicted_price - current_price):.2f} ({abs((final_predicted_price - current_price) / current_price * 100):.2f}%)",
                        delta_color="normal" if final_predicted_price >= current_price else "inverse"
                    )
                    st.caption(f"Prediction Confidence: {confidence:.2f}%")
                    st.caption(f"Model learns from each prediction")
                
                # Create a DataFrame for the 7-day prediction table
                prediction_dates = [date.strftime('%d-%m-%Y') for date, _ in prediction_results]
                prediction_prices = [f"₹{price:.2f}" for _, price in prediction_results]
                
                predictions_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted Price': prediction_prices
                })
                
                # Display the prediction table
                st.subheader("Daily Price Predictions")
                st.table(predictions_df)
                
                # Display sentiment analysis
                st.subheader("Market Sentiment")
                sentiment_col1, sentiment_col2 = st.columns(2)
                
                with sentiment_col1:
                    sentiment_score = news_sentiment['compound_score']
                    sentiment_label = "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral"
                    sentiment_color = "green" if sentiment_score > 0.1 else "red" if sentiment_score < -0.1 else "gray"
                    
                    st.markdown(f"<h3 style='color: {sentiment_color}; margin-bottom: 0;'>{sentiment_label}</h3>", unsafe_allow_html=True)
                    st.caption(f"Sentiment Score: {sentiment_score:.2f} (Range: -1 to +1)")
                    
                with sentiment_col2:
                    st.caption("Based on analysis of recent news:")
                    for headline in news_sentiment['headlines'][:3]:
                        st.markdown(f"• {headline}")
                
                # Plot historical data and prediction
                st.subheader("Price History & Prediction")
                
                # Create the plotly figure
                fig = go.Figure()
                
                # Add historical data line
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Extract dates and predicted values for plotting
                today = end_date
                if not hasattr(today, 'tzinfo') or today.tzinfo is None:
                    today = today.replace(tzinfo=pytz.UTC)
                
                # Add today's price to create a continuous line
                plot_dates = [today] + [date for date, _ in prediction_results]
                plot_values = [historical_data['Close'].iloc[-1]] + [price for _, price in prediction_results]
                
                # Add the 7-day prediction line
                fig.add_trace(go.Scatter(
                    x=plot_dates,
                    y=plot_values,
                    mode='lines+markers',
                    name='7-Day Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(
                        color='red',
                        size=[5] + [5] * 6 + [10],  # Make today's and the final point larger
                    )
                ))
                    
                # No need to add the additional prediction trend line
                
                # Update layout for a cleaner look
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display technical indicators section
                with st.expander("Technical Indicators"):
                    # Calculate some basic technical indicators
                    tech_data = historical_data.copy()
                    
                    # 50-day and 200-day Moving Averages
                    tech_data['MA50'] = tech_data['Close'].rolling(window=50).mean()
                    tech_data['MA200'] = tech_data['Close'].rolling(window=200).mean()
                    
                    # Relative Strength Index (RSI) - simplified
                    delta = tech_data['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    tech_data['RSI'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    tech_data['EMA12'] = tech_data['Close'].ewm(span=12, adjust=False).mean()
                    tech_data['EMA26'] = tech_data['Close'].ewm(span=26, adjust=False).mean()
                    tech_data['MACD'] = tech_data['EMA12'] - tech_data['EMA26']
                    tech_data['Signal'] = tech_data['MACD'].ewm(span=9, adjust=False).mean()
                    
                    # Display the indicators in a clean format
                    recent_data = tech_data.iloc[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("50-Day MA", f"₹{format_price(recent_data['MA50'])}")
                        st.metric("200-Day MA", f"₹{format_price(recent_data['MA200'])}")
                    
                    with col2:
                        rsi_value = recent_data['RSI']
                        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        st.metric("RSI (14)", f"{rsi_value:.2f}")
                        st.caption(f"Status: {rsi_status}")
                    
                    with col3:
                        macd = recent_data['MACD']
                        signal = recent_data['Signal']
                        macd_diff = macd - signal
                        macd_status = "Bullish" if macd_diff > 0 else "Bearish"
                        
                        st.metric("MACD", f"{macd:.2f}")
                        st.caption(f"Signal: {signal:.2f} ({macd_status})")
            else:
                st.error("Could not retrieve historical data for this stock.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.caption("Please check the stock symbol and try again.")

# Footer
st.markdown("---")
st.caption("This app uses real-time data from Yahoo Finance. Predictions are based on historical trends and sentiment analysis of news articles.")
st.caption("⚠️ Disclaimer: Stock predictions are for informational purposes only. Always do your own research before making investment decisions.")
