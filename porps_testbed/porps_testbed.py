import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

i = 0.0002      # Daily risk-free rate

class Portfolio:

    # Prices for AAPL, SBUX, MSFT are pulled from https://www.nasdaq.com/market-activity/quotes/historical.
    # Data spans 152 days.

    try: 
        d_aapl = pd.read_csv("data/apple.csv")
        d_sbux = pd.read_csv("data/starbucks.csv")
        d_msft = pd.read_csv("data/microsoft.csv")
    except FileNotFoundError:
        print("Error: CSV files not found.")
        exit()

    # Stock prices start at the present time (index 0) to the oldest time (index 152).
    # Use [::-1].reset_index(drop=True) to flip the columns of the closing prices upside down.

    dates = d_aapl['Date'][::-1].reset_index(drop=True)
    close_aapl = d_aapl['Close/Last'][::-1].reset_index(drop=True).str.replace('$', '', regex=False).astype(float)
    close_sbux = d_sbux['Close/Last'][::-1].reset_index(drop=True).str.replace('$', '', regex=False).astype(float)
    close_msft = d_msft['Close/Last'][::-1].reset_index(drop=True).str.replace('$', '', regex=False).astype(float)

    # DataFrame initialization

    data = {'Date' : dates, 'AAPL' : close_aapl, 'SBUX' : close_sbux, 'MSFT' : close_msft}
    df = pd.DataFrame(data)
    print(df['AAPL'])
    # Daily returns

    df['dyret_aapl'] = df['AAPL'].pct_change()
    df['dyret_sbux'] = df['SBUX'].pct_change()
    df['dyret_msft'] = df['MSFT'].pct_change()

    # Average returns

    avret_aapl = df['dyret_aapl'].mean()*np.sqrt(152)
    avret_sbux = df['dyret_sbux'].mean()*np.sqrt(152)
    avret_msft = df['dyret_msft'].mean()*np.sqrt(152)

    # Covariance matrices

    cov_aaplsbux = df['dyret_aapl'].cov(df['dyret_sbux'])*np.sqrt(152)
    cov_aaplmsft = df['dyret_aapl'].cov(df['dyret_msft'])*np.sqrt(152)
    cov_sbuxmsft = df['dyret_sbux'].cov(df['dyret_msft'])*np.sqrt(152)


def portfolio_metrics(weights, *args, **kwargs):
    # This function evaluates portfolio performance for a set of weights
    # Supported metrics are return, volatility, and the Sharpe Ratio
    # Stores results in a dictionary for the efficient frontier calculation
    return 0