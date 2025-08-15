import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        print(self.df['AAPL'])
        # Daily returns
         
        self.df['dyret_aapl'] = self.df['AAPL'].pct_change()
        self.df['dyret_sbux'] = self.df['SBUX'].pct_change()
        self.df['dyret_msft'] = self.df['MSFT'].pct_change()

        # Instance attributes
        self.df = self.df.dropna()
        self.mean_returns = self.df[['dyret_aapl', 'dyret_sbux', 'dyret_msft']].mean()*152
        self.cov_matrix = self.df[['dyret_aapl', 'dyret_sbux', 'dyret_msft']].cov() * 152
        self.risk_free_rate = i*152

    def portfolio_metrics(self, weights, *args, **kwargs):
    # This function evaluates portfolio performance for a set of weights
    # Supported metrics are return, volatility, and the Sharpe Ratio
    # Stores results in a dictionary for the efficient frontier calculation
        
        risk_free_rate = kwargs.get('risk_free_rate', self.risk_free_rate)
        store = kwargs.get('store', False)                                          # If false, return single metric
        ptfr_id = kwargs.get('ptfrid', None)                                        # Assign an ID to the portfolio 

        annret_prtf = np.sum(weights * self.mean_returns)                           # Annualized portfolio return
        annvol_prtf = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))  # Annualized portfolio volatility
        sharpe = (annret_prtf - risk_free_rate)/annvol_prtf                         # Portfolio Sharpe ratio

    # Return Sharpe ratio, return, or volatility depending on user input in args
        metric = args[0] if args else 'sharpe'

    # If True is passed in **kwargs when calling the function, the entire dictionary of metrics is returned
        if store == True:
            return {'ptfrid': ptfr_id if ptfr_id is not None else 'default', 'return': annret_prtf, 'volatility': annvol_prtf, 'sharpe': sharpe}
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
        w_start = np.ones(n)/n # Starting guess [1, 1, 1, ..., 1]
        bounds = tuple((-1, 1) for _ in range(n)) # Bounds on portfolio weights
        main_constraint = {'type' : 'eq', 'fun': lambda w: np.sum(w) - 1} # The type of contraint is equality where the sum of weights must equal zero
        

    # Optimization based on specified target return (maximize Sharpe ratio or meet target return)
        if target_return == None:
            result = minimize(lambda w: -self.portfolio_metrics(w, 'sharpe'), w_start, method = 'SLSQP', bounds = bounds, constraints = [main_constraint])
        else:
            return_constraint =  {'type' : 'eq', 'fun' : lambda w: self.portfolio_metrics(w, 'return') - target_return}
            result = minimize(lambda w: -self.portfolio_metrics(w, 'volatility'), w_start, method = 'SLSQP', bounds = bounds, constraints = [main_constraint, return_constraint])
            
    # Check if optimization succeeded 
        if result.success:
            return {'optimized weights' : result.x, 'portfolio metrics' : self.portfolio_metrics(result.x, store = True)}
        else:
            raise ValueError(result.message)

    def generate_portfolios(self, num_portfolios = 1000):
        np.random.dirichlet(np.ones(n), num_portfolios)
        return 0;
        
    def predict_volatility(self):
        return 0;
       


