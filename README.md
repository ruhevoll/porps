# Portfolio Optimization and Risk Prediction System (PORPS)

<img width="986" height="579" alt="image" src="https://github.com/user-attachments/assets/2dedf997-412b-405a-a44e-3d58f873aa07" />
<img width="986" height="579" alt="image" src="https://github.com/user-attachments/assets/ec4410f7-988f-4a57-a869-8b92f8ab0519" />

## What is this code?
This code uses 152 days of market data from Apple, Starbucks, and Microsoft to compute the optimal portfolio weights for a portfolio including these 3 companies. 
Portfolio optimization may be done in two ways: By default, it will find the portfolio with the maximum Sharpe ratio. However, the user can also specify a desired return and the program will compute the portfolio yielding that return with minimum volatility.

Moreover, this program includes a model trained on the first 80% of the market data to predict the volatility in the remaining 20%, judging by the pictures above, it's not that bad!

# Why did I do this?
The purpose of this project is for my personal review of Python, the pandas/numpy/matplotlib libraries, and to help me learn how to implement machine learning in Python via scikit-learn. 

## Future work
I'd like to make this code more universal, so you can dynamically add stocks and consider a risk-free account as a part of your portfolio. Right now, values are somewhat hard coded for AAPL, SBUX, and MSFT, and it's also hard coded for a trading period of 152 days. Also, some more detailed documentation explaining the mathematics behind this code. 
