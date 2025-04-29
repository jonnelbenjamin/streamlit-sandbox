import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

#------------------------------
# Load Sample Dataset
#------------------------------
def load_sample_data():
    """
    Load a sample dataset (AirPassengers) for demonstration purposes.
    The dataset contains monthly airline passenger counts from 1949 to 1960.
    - The data is resampled to ensure monthly frequency.
    - Returns a pandas Series with the passenger counts.
    """
    from statsmodels.datasets import get_rdataset
    data = get_rdataset('AirPassengers')
    df = data.data
    df['date'] = pd.to_datetime(df['time'].astype(str).str.replace('.0', '-01-01'), format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    df = df.resample('M').sum()  # Resample to monthly frequency
    return df['value'].rename('passengers')

#------------------------------
# Stationarity Test
#------------------------------
def test_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    - Stationarity means the statistical properties (mean, variance) do not change over time.
    - ADF test outputs a p-value; if p < 0.05, the series is stationary.
    """
    adf_result = adfuller(series.dropna())
    p_value = adf_result[1]
    st.write("### Stationarity Test (ADF)")
    st.write(f"Test Statistic: {adf_result[0]:.4f}")
    st.write(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        st.success("Series is stationary (p < 0.05)")
    else:
        st.warning("Series is non-stationary (p >= 0.05) - Differencing or transformation needed")

#------------------------------
# Decomposition Plot
#------------------------------
def plot_decomposition(series, model='additive', period=12):
    """
    Decompose the time series into its components:
    - Observed: Original data
    - Trend: Long-term movement
    - Seasonal: Repeating patterns
    - Residual: Random noise
    - The decomposition helps understand the structure of the data.
    """
    decomposition = seasonal_decompose(series, model=model, period=period)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residuals')
    plt.tight_layout()
    return fig

#------------------------------
# Evaluation Metrics
#------------------------------
def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the forecasting model using various metrics:
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of Determination
    - MAPE: Mean Absolute Percentage Error
    - SMAPE: Symmetric Mean Absolute Percentage Error
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE', 'R²', 'MAPE', 'SMAPE'],
        'Value': [mse, mae, rmse, r2, mape, smape]
    })

#------------------------------
# Main App
#------------------------------
def time_series_analyzer():
    """
    Main function to run the Streamlit app for time series analysis and forecasting.
    - Provides options for data upload or using a sample dataset.
    - Performs EDA, stationarity tests, and model building (ARIMA, SARIMA, Prophet).
    - Visualizes results and evaluates model performance.
    """
    st.title("📈 Time Series Analysis and Forecasting")
    
    # Sidebar for data options
    st.sidebar.header("Data Options")
    use_sample = st.sidebar.checkbox("Use sample data (AirPassengers)", True)
    
    if use_sample:
        # Load and display sample dataset
        series = load_sample_data()
        st.write("### Sample Dataset: Air Passengers (Monthly, 1949-1960)")
        st.line_chart(series)
    else:
        # Allow user to upload their own dataset
        uploaded = st.sidebar.file_uploader("Upload your CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            date_col = st.sidebar.selectbox("Date column", df.columns)
            value_col = st.sidebar.selectbox("Value column", df.columns)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            series = df[value_col]
            st.write(f"### Uploaded Data: {value_col}")
            st.line_chart(series)
        else:
            st.stop()
    
    # Exploratory Data Analysis (EDA)
    st.header("1. 📊 Exploratory Data Analysis (EDA)")
    st.write("""
    Time Series data often has the following components:
    - **Trend**: Long-term upward or downward movement.
    - **Seasonality**: Regular patterns that repeat at fixed periods (e.g., every year).
    - **Cyclicality**: Long-term oscillations (different from seasonality).
    - **Residual/Noise**: Random fluctuations.

    We first visualize and decompose the time series.
    """)
    decomposition_model = st.radio("Decomposition model type", ['additive', 'multiplicative'])
    decomp_fig = plot_decomposition(series, model=decomposition_model)
    st.pyplot(decomp_fig)
    
    # Autocorrelation Analysis
    st.write("### Autocorrelation Analysis")
    st.write("""
    - **ACF (Autocorrelation Function)**: Measures how the current value depends on past values.
    - **PACF (Partial ACF)**: Measures the correlation controlling for intermediate lags.
    Useful for choosing AR (p) and MA (q) terms.
    """)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=24, ax=ax1)
    plot_pacf(series, lags=24, method='ywm', ax=ax2)
    st.pyplot(fig)
    
    # Stationarity Test
    st.header("2. 🧪 Stationarity Test")
    st.write("""
    Stationarity means statistical properties like mean and variance are constant over time.
    Most forecasting models (ARIMA, SARIMA) **require** stationary series.
    """)
    test_stationarity(series)
    
    # Model Building and Forecasting
    st.header("3. ⚙️ Model Building and Forecasting")
    model_type = st.selectbox("Choose Model", ["ARIMA", "SARIMA", "Prophet"])
    
    # Train/Test Split
    split_ratio = st.slider("Training data (%)", 50, 95, 80)
    split_idx = int(len(series) * (split_ratio/100))
    train, test = series[:split_idx], series[split_idx:]
    
    if model_type == "ARIMA":
        # ARIMA Model
        st.subheader("ARIMA Model")
        st.write("""
        ARIMA (AutoRegressive Integrated Moving Average):
        - p = Autoregressive lags
        - d = Differencing to make stationary
        - q = Moving average window size
        """)
        col1, col2, col3 = st.columns(3)
        p = col1.number_input("AR Order (p)", 0, 5, 2)
        d = col2.number_input("Differencing Order (d)", 0, 2, 1)
        q = col3.number_input("MA Order (q)", 0, 5, 2)
        
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        st.write(model_fit.summary())
        
        pred = model_fit.forecast(steps=len(test))
        pred.index = test.index
        
    elif model_type == "SARIMA":
        # SARIMA Model
        st.subheader("SARIMA Model (Seasonal ARIMA)")
        st.write("""
        SARIMA extends ARIMA by adding **seasonal components**:
        - P = Seasonal autoregressive order
        - D = Seasonal differencing
        - Q = Seasonal moving average
        - s = Seasonal period (e.g., 12 months for yearly)
        """)
        col1, col2, col3 = st.columns(3)
        p = col1.number_input("AR order (p)", 0, 5, 1)
        d = col2.number_input("Differencing (d)", 0, 2, 1)
        q = col3.number_input("MA order (q)", 0, 5, 1)
        
        col4, col5, col6 = st.columns(3)
        P = col4.number_input("Seasonal AR (P)", 0, 5, 1)
        D = col5.number_input("Seasonal differencing (D)", 0, 2, 1)
        Q = col6.number_input("Seasonal MA (Q)", 0, 5, 1)
        s = st.number_input("Season length (s)", 1, 24, 12)
        
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit()
        st.write(model_fit.summary())
        
        pred = model_fit.forecast(steps=len(test))
        pred.index = test.index
        
    elif model_type == "Prophet":
        # Prophet Model
        st.subheader("Prophet Model (by Meta/Facebook)")
        st.write("""
        Prophet is robust to:
        - Missing data
        - Shifts in seasonality
        - Holiday effects
        Best for business and web applications.
        """)
        
        prophet_df = train.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=decomposition_model
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=len(test), freq='M')
        forecast = model.predict(future)
        pred = forecast.set_index('ds')['yhat'][-len(test):]
    
    # Plot forecasts
    st.subheader("Forecast vs Actuals")
    fig, ax = plt.subplots(figsize=(12,6))
    train.plot(ax=ax, label="Training")
    test.plot(ax=ax, label="Actual", color='black')
    pred.plot(ax=ax, label="Forecast", color='red')
    plt.legend()
    st.pyplot(fig)
    
    # Model Performance
    st.subheader("📈 Model Performance")
    metrics = evaluate_model(test, pred)
    st.table(metrics)

#------------------------------
# Run the App
#------------------------------
if __name__ == "__main__":
    time_series_analyzer()