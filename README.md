# PyRPO

Markowitz Mean-Variance Optimization is highly sensitive to small estimation errors in parameters. This problem makes the method unsuable in practice. To counteract these problems, portfolio optimization can factor in uncertainty when estimating parameters. The optimization problem becomes 

$$
\begin{align}
\max \text{ } & \mathbf{w}^{T} \mathbf{\mu} + \delta \sqrt{\mathbf{w}^{T} \mathbf{\Sigma}_{\mu} \mathbf{w}} - \gamma \mathbf{w}^{T} \mathbf{\Sigma} \mathbf{w}\\
\text{s.t. } & \mathbf{w}^{T}\mathbf{1} = 1
\end{align}
$$

where $\mathbf{\mu}$ is a vector of expected returns, $\mathbf{w}$ are the asset proportions/weights, $\delta$ is the uncertainty radius, $\mathbf{\Sigma}$, is the variance-covariance matrix of returns, and $\mathbf{\Sigma}_{\mu}$ is a diagonal matrix of returns variances. Note that we are technically trying to maximize _risk-adjusted returns_ since we have a risk aversion term. One may simply set $\gamma = 0$ to ignore risk-adjustment.

`PyRPO` is a Python package that implements Robust Portfolio Optimization. It uses an ellipsoidal uncertainty set for robust optimization.

## Installation
You can install the package using pip:
```
pip install git+https://github.com/datstat-consulting/PyRPO
```

# Example
We demonstrate the library using cryptocurrency as an example. This shows how versatile the library is with any kind of Financial asset data.

Obtain cryptocurrency closing prices.
```
exchange = ccxt.bitstamp()
closing_prices = pd.DataFrame()

symbols = ['XRP/USD', 'ETH/USD', 'ADA/USD']
timeframe = '1d'
start_date = '2022-12-01T00:00:00Z'

for symbol in symbols:
  since = exchange.parse8601(start_date)
  data = []
  while True:
    candles = exchange.fetch_ohlcv(symbol, timeframe, since)
    if not candles:
      break
    data.extend(candles)
    since = candles[-1][0] + 1

  df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
  df.set_index('timestamp', inplace=True)
  closing_prices[symbol] = df['close']
  
closing_prices.to_csv('PortfolioData.csv')
```
Import the class, and create an instance of the class with a sample CSV file containing historical price data.
```
from PyRPO import *
instanceRPO = PyRPO('PortfolioData.csv')
train_data, test_data = instanceRPO.train_test_split()
```
Set the risk aversion parameter (gamma) and solve the RPO problem. When no uncertainty radius is set, PyRPO uses half the average Sharpe Ratios as the Uncertainty Radius as suggested by Yin, Perchet, and Soupé (2021). When no time period is set, PyRPO uses daily returns at 252 working days.
```
gamma = 0.5
#uncertainty_radius = 0.1
instanceRPO.solve_rpo(gamma, uncertainty_radius = None, risk_free_rate = 0.05, time_period = 360)

print("Optimal weights:", instanceRPO.optimal_weights)
```
Perform sensitivity analysis with a range of uncertainty factors.
```
uncertainty_bounds_factor_range = np.linspace(0.9, 1.1, 50)
instanceRPO.sensitivity_analysis(uncertainty_bounds_factor_range, gamma)
```
Plot the optimal weights and sensitivity analyses using matplotlib.
```
instanceRPO.plot_optimal_weights(risk_free_rate = 0.05, time_period = 360)
instanceRPO.plot_sensitivity_analysis(uncertainty_bounds_factor_range)
```
Generate the same plots using Plotly.
```
import plotly.io as pio

optimal_weights_figure = instanceRPO.generate_optimal_weights_figure(risk_free_rate = 0.05, time_period = 360)
pio.show(optimal_weights_figure)
sensitivity_analysis_figure = instanceRPO.generate_sensitivity_analysis_figure(uncertainty_bounds_factor_range)
pio.show(sensitivity_analysis_figure)
```
Perform backtesting.
```
# Evaluate the performance on the testing dataset
test_daily_returns = test_data.dot(instanceRPO.optimal_weights)
sharpe_ratio = instanceRPO.sharpe_ratio(test_daily_returns)

print("Sharpe Ratio:", sharpe_ratio)

test_daily_returns, test_cumulative_returns, equally_weighted_daily_returns, equally_weighted_cumulative_returns = instanceRPO.backtesting(test_data=test_data)

print("Test Cumulative Returns (Optimal Portfolio):", test_cumulative_returns)
print("Test Cumulative Returns (Equally-Weighted Portfolio):", equally_weighted_cumulative_returns)
```
Plot backtesting results.
```
instanceRPO.plot_backtesting(test_cumulative_returns, equally_weighted_cumulative_returns)

backtesting_figure = instanceRPO.generate_backtesting_figure(test_cumulative_returns, equally_weighted_cumulative_returns)
backtesting_figure.show()
```
Allocate capital.
```
instanceRPO.allocate_capital(54368)
instanceRPO.generate_capital_allocation_chart()
print(instanceRPO.capital_allocation)
```
# References
* Feng, Y., & Palomar, D. P. (2016). A signal processing perspective on financial engineering. Foundations and Trends® in Signal Processing, 9(1–2), 1-231.
* Georgantas, A., Doumpos, M., & Zopounidis, C. (2021). Robust optimization approaches for portfolio selection: a comparative analysis. Annals of Operations Research, 1-17.
* Yin, C., Perchet, R., & Soupé, F. (2021). A practical guide to robust portfolio optimization. Quantitative Finance, 21(6), 911-928.
