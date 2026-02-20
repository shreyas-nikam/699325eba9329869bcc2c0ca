import yfinance as yf
import pandas as pd
import numpy as np

from source import acquire_and_preprocess_market_data

# Execute the data acquisition and preprocessing
market_returns, known_anomaly_dates = acquire_and_preprocess_market_data()
