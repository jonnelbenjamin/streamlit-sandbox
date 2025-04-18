import streamlit as st
import pandas as pd
import prophet
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import yfinance as yf
import zipfile
import os


# Function to load the CSV file
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def plot_metrics(forecast, df):
    # Plot the key metrics: actual vs predicted
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(forecast['ds'], df['y'], label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_title("Forecast vs Actuals")
    ax.legend()
    st.pyplot(fig)

def run_forecasting():
    st.title("Prophet Time Series Forecasting")

    st.write(
        "Prophet is a forecasting tool used for time series data, ideal for scenarios with daily, weekly, or yearly trends. "
        "It can model seasonal effects, holidays, and other irregularities in data. Common business use cases include:\n\n"
        "- **Demand Forecasting**: Predict future product demand for inventory planning.\n"
        "- **Sales Forecasting**: Forecast future sales based on historical sales data.\n"
        "- **Financial Forecasting**: Predict stock prices or revenues based on historical trends.\n"
    )

    # 1. **Sales Forecasting** - Retail Sales
    st.subheader("Sales Forecasting")
    retail_file_path = './components/datasets/Online_Retail.xlsx'  # Path to the zip file
  
    df = load_data(retail_file_path)

    # Prepare data for Prophet (Assume 'date' and 'value' columns for time series)
    df['ds'] = pd.to_datetime(df['InvoiceDate'])
    df['y'] = df['UnitPrice']

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe( periods=365)
    forecast = model.predict(future)

    # Plot the forecast and confidence intervals
    st.subheader("Sales Forecast with Confidence Intervals")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    plot_metrics(forecast, df)

    # Show trend and seasonal components
    st.subheader("Sales Trend and Seasonal Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Explain Sales Forecasting Metrics
    st.subheader("Sales Forecasting Metrics Explained")
    st.write(
        "1. **`yhat`**: The predicted future sales. It shows the model's forecast based on the historical data.\n"
        "2. **`yhat_lower` and `yhat_upper`**: These are the confidence intervals around the forecast (80% by default). "
        "It shows the expected range of values within which the actual sales are likely to fall.\n"
        "3. **Seasonality**: Prophet captures any repeating seasonal patterns in the data. For example, retail businesses may have higher sales during holidays."
    )

    # 2. **Demand Forecasting** - Hourly Demand Data
    st.subheader("Demand Forecasting")
    # demand_data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"  # replace with your dataset
    # df_demand = load_data(demand_data_url)
    # df_demand.columns = ['ds', 'y']
    # df_demand['ds'] = pd.to_datetime(df_demand['ds'])

    # model.fit(df_demand)
    # future_demand = model.make_future_dataframe(df_demand, periods=12)
    # forecast_demand = model.predict(future_demand)

    # Plot the forecast and confidence intervals for demand
    # st.subheader("Demand Forecast with Confidence Intervals")
    # fig3 = model.plot(forecast_demand)
    # st.pyplot(fig3)
    # plot_metrics(forecast_demand)

    # Show trend and seasonal components for demand
    # st.subheader("Demand Trend and Seasonal Components")
    # fig4 = model.plot_components(forecast_demand)
    # st.pyplot(fig4)

    # Explain Demand Forecasting Metrics
    # st.subheader("Demand Forecasting Metrics Explained")
    # st.write(
    #     "1. **`yhat`**: The predicted future demand for a product (e.g., number of passengers). This helps businesses plan production and inventory.\n"
    #     "2. **`yhat_lower` and `yhat_upper`**: These are the 80% confidence intervals around the demand forecast. "
    #     "A wider interval may indicate uncertainty, while a narrower interval suggests more certainty.\n"
    #     "3. **Trend**: Shows whether the demand is increasing or decreasing over time, which is useful for long-term planning.\n"
    #     "4. **Seasonality**: Captures regular patterns in the demand, such as seasonal fluctuations in airline passengers during peak travel seasons."
    # )

    # 3. **Financial Forecasting** - Stock Prices using Yahoo Finance
    # st.subheader("Financial Forecasting")
    # stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL")
    
    # Download stock data from Yahoo Finance (past 3 years)
    # stock_data = yf.download(stock_ticker, period="3y", interval="1d")
    # stock_data.reset_index(inplace=True)
    # df_stock = stock_data[['Date', 'Close']]
    # df_stock.columns = ['ds', 'y']
    # df_stock['ds'] = pd.to_datetime(df_stock['ds'])

    # model.fit(df_stock)
    # future_stock = model.make_future_dataframe(df_stock, periods=30)
    # forecast_stock = model.predict(future_stock)

    # Plot the forecast and confidence intervals for financial forecasting
    # st.subheader(f"{stock_ticker} Stock Price Forecast with Confidence Intervals")
    # fig5 = model.plot(forecast_stock)
    # st.pyplot(fig5)
    # plot_metrics(forecast_stock)

    # Show trend and seasonal components for stock data
    # st.subheader(f"{stock_ticker} Trend and Seasonal Components")
    # fig6 = model.plot_components(forecast_stock)
    # st.pyplot(fig6)

    # Explain Financial Forecasting Metrics
    # st.subheader("Financial Forecasting Metrics Explained")
    # st.write(
    #     "1. **`yhat`**: The predicted future stock price. It indicates the expected value of the stock at a given time in the future.\n"
    #     "2. **`yhat_lower` and `yhat_upper`**: These are the confidence intervals. A narrow range indicates a high level of confidence, "
    #     "while a wider range suggests more volatility or uncertainty in the prediction.\n"
    #     "3. **Trend**: Shows the overall direction of the stock price (upward or downward) over time.\n"
    #     "4. **Seasonality**: Captures any recurring patterns or cycles, such as quarterly earnings cycles or market reactions to specific events."
    # )

    # Add a business use case section
    st.subheader("Business Use Case: Sales Forecasting")
    st.write(
        "In sales forecasting, Prophet helps predict future sales based on historical data. "
        "This is useful for inventory management, sales targets, and budgeting.\n"
        "Here, the model demonstrates how future sales can be predicted based on past data."
    )

if __name__ == "__main__":
    run_forecasting()
