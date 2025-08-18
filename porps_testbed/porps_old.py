# © Jacob White 2025 All Rights Reserved

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



i = 0.0002      # Daily risk-free rate

class Portfolio:
    def __init__(self): 
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
        self.df = pd.DataFrame(data)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        print(self.df['AAPL'])
        # Daily returns
         
        self.df['dyret_aapl'] = self.df['AAPL'].pct_change()
        self.df['dyret_sbux'] = self.df['SBUX'].pct_change()
        self.df['dyret_msft'] = self.df['MSFT'].pct_change()

        # Instance attributes
        self.df = self.df.dropna()
        self.mean_returns = self.df[['dyret_aapl', 'dyret_sbux', 'dyret_msft']].mean()*np.sqrt(152)
        self.cov_matrix = self.df[['dyret_aapl', 'dyret_sbux', 'dyret_msft']].cov() * 152
        self.risk_free_rate = i*152
        print("Mean Returns:", self.mean_returns)
        print("Cov Matrix:", self.cov_matrix)
    def portfolio_metrics(self, weights, *args, **kwargs):
    # This function evaluates portfolio performance for a set of weights
    # Supported metrics are return, volatility, and the Sharpe Ratio
    # Stores results in a dictionary for the efficient frontier calculation
        
        risk_free_rate = kwargs.get('risk_free_rate', self.risk_free_rate)
        store = kwargs.get('store', False)                                          # If false, return single metric
        prtfid = kwargs.get('prtfid', None)                                        # Assign an ID to the portfolio 

        annret_prtf = np.sum(weights * self.mean_returns * np.sqrt(152))                           # Annualized portfolio return
        annvol_prtf = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))  # Annualized portfolio volatility
        sharpe = (annret_prtf - risk_free_rate)/annvol_prtf                         # Portfolio Sharpe ratio

    # Return Sharpe ratio, return, or volatility depending on user input in args
        metric = args[0] if args else 'sharpe'

    # If True is passed in **kwargs when calling the function, the entire dictionary of metrics is returned
        if store == True:
            return {'prtfid': prtfid if prtfid is not None else 'default', 'return': annret_prtf, 'volatility': annvol_prtf, 'sharpe': sharpe}
        elif store == False:
            if metric == 'return':
                return annret_prtf
            elif metric == 'volatility':
                return annvol_prtf
            elif metric == 'sharpe':
                return sharpe
            else:
               raise ValueError("Invalid metric type, valid metric types are 'return', 'volatility', and 'sharpe'.")

    def optimize_portfolio(self, target_return = None):

        n = len(self.mean_returns) # Number of stocks, doing it this way allows for scalability later!
        w_start = np.ones(n)/n # Starting guess, vector that sums to 1
        bounds = tuple((0, 1) for _ in range(n)) # Bounds on portfolio weights
        main_constraint = {'type' : 'eq', 'fun': lambda w: np.sum(w) - 1} # The type of contraint is equality where the sum of weights must equal zero
        

    # Optimization based on specified target return (maximize Sharpe ratio or meet target return)
        if target_return == None:
            result = minimize(lambda w: -self.portfolio_metrics(w, 'sharpe'), w_start, method = 'SLSQP', bounds = bounds, constraints = [main_constraint])
        else:
            return_constraint =  {'type' : 'eq', 'fun' : lambda w: self.portfolio_metrics(w, 'return') - target_return}
            result = minimize(lambda w: self.portfolio_metrics(w, 'volatility'), w_start, method = 'SLSQP', bounds = bounds, constraints = [main_constraint, return_constraint])
            
    # Check if optimization succeeded 
        if result.success:
            print("Optimized Weights:", result.x)
            return {'optimized weights' : result.x, 'portfolio metrics' : self.portfolio_metrics(result.x, store = True)}
        else:
            raise ValueError(result.message)

    def generate_portfolios(self, num_portfolios = 1000):
        # Generate portfolios and store their metrics in a dictionary

        n = len(self.mean_returns)
        weights = np.random.dirichlet(np.ones(n), num_portfolios) # Generate n weights for num_portfolios number of portfolios, [[0.2, 0.4, 0.4], [0.3, 0.3, 0.4], ...] etc.
        portfolios = {}
        print('Weights: \n')
        print(weights)
        # Here, we iterate over the portfolios and store their metrics in a dictionary

        for idx in range(0, num_portfolios):
            portfolios[f'port{idx}'] = self.portfolio_metrics(weights[idx], store = True, prtfid = f'port{idx}')
        return portfolios


    def predict_volatility(self):
        # Computes rolling volatility, and predicts future volatility (wouldn't ship this, this is just a first pass at trying this)
        # USING TEST-SET VALIDATION (see ISLP 5.1) w/ first 80% training data and 20% test data
        self.df['ret_prtf'] = (self.df[['dyret_aapl', 'dyret_sbux', 'dyret_msft']] * [1/3, 1/3, 1/3]).sum(axis=1)
        self.df['vol_prtf'] = self.df['ret_prtf'].rolling(window = 21).std() * np.sqrt(152) # Rolling standard deviation in 21-day window

        # Lagged returns & lagged vol. to predict volatility
        self.df['lag_aapl'] = self.df['dyret_aapl'].shift(1)
        self.df['lag_sbux'] = self.df['dyret_sbux'].shift(1)
        self.df['lag_msft'] = self.df['dyret_msft'].shift(1)
        self.df['lag_vol'] = self.df['vol_prtf'].shift(1)
        data = self.df[['Date', 'vol_prtf', 'lag_aapl', 'lag_sbux', 'lag_msft', 'lag_vol']].dropna()
       
        training_size = int(0.8*len(data))
        X_train = data.iloc[:training_size][['lag_aapl', 'lag_sbux', 'lag_msft', 'lag_vol']] #iloc is integer location, selects specific entries up to the training size
        Y_train = data.iloc[:training_size]['vol_prtf']
        X_test = data.iloc[training_size:][['lag_aapl', 'lag_sbux', 'lag_msft', 'lag_vol']]
        Y_test = data.iloc[training_size:]['vol_prtf']

        # Training process, ft. feature scaling

        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)

        return {'predicted' : Y_pred, 'actual' : Y_test, 'dates' : data.iloc[training_size:]['Date']}

    def plot_results(self, prtf_metrics_dict):
        
        # Efficient frontier plot
        plt.figure(figsize=(10, 6))
        returns = [prtf_metrics_dict[key]['return'] for key in prtf_metrics_dict]
        volatilities = [prtf_metrics_dict[key]['volatility'] for key in prtf_metrics_dict]
        plt.scatter(volatilities, returns, c='blue', alpha=0.5, label='Portfolios')
        opt_result = self.optimize_portfolio()['portfolio metrics']
        plt.scatter(opt_result['volatility'], opt_result['return'], c='red', marker='*', s=200, label='Optimal Portfolio')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Volatility prediction plot
        plt.figure(figsize=(10, 6))
        vol_results = self.predict_volatility()
        predicted = vol_results['predicted']
        actual = vol_results['actual']
        dates = pd.to_datetime(vol_results['dates'])
        plt.plot(dates, predicted, label='Predicted Volatility', color='blue')
        plt.plot(dates, actual, label='Actual Volatility', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.title('Predicted vs. Actual Portfolio Volatility')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()