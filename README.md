# PyRPO

Markowitz Mean-Variance Optimization is highly sensitive to small estimation errors in parameters. This problem makes the method unsuable in practice. To counteract these problems, portfolio optimization can factor in uncertainty when estimating parameters.

`PyRPO` is a Python package that implements Robust Portfolio Optimization. This library was built as a hobby project. `PyRPO` relies heavily on numpy and scipy to handle matrix algebra and optimization. 

# Example

Obtain cryptocurrency closing prices.
```
exchange = ccxt.bitstamp()
closing_prices = pd.DataFrame()

symbols = ['BTC/USD', 'ETH/USD', 'GALA/USD']
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
Create an instance of the class with a sample CSV file containing historical price data.
```
instanceRPO = RobustPortfolioOptimizer('PortfolioData.csv')
```
Set the risk aversion parameter (gamma) and solve the RPO problem. When no uncertainty radius is set, PyRPO uses half the average Sharpe Ratios as the Uncertainty Radius as suggested by Yin, Perchet, and Soupé (2021).
```
gamma = 0.5
#uncertainty_radius = 0.1
instanceRPO.solve_rpo(gamma, uncertainty_radius = None)

print("Optimal weights:", optimizer.optimal_weights)
```
Perform sensitivity analysis with a range of uncertainty factors.
```
uncertainty_bounds_factor_range = np.linspace(0.9, 1.1, 50)
instanceRPO.sensitivity_analysis(uncertainty_bounds_factor_range)
```
Plot the optimal weights and sensitivity analyses using matplotlib.
```
instanceRPO.plot_optimal_weights()
instanceRPO.plot_sensitivity_analysis(uncertainty_bounds_factor_range)
```
Generate the same plots using Plotly.
```
import plotly.io as pio

optimal_weights_figure = instanceRPO.generate_optimal_weights_figure()
pio.show(optimal_weights_figure)
sensitivity_analysis_figure = instanceRPO.generate_sensitivity_analysis_figure(uncertainty_bounds_factor_range)
pio.show(sensitivity_analysis_figure)
```
Perform backtesting, and plot the results.
```
instanceRPO.backtesting()
instanceRPO.plot_backtesting()
backtesting_figure = instanceRPO.generate_backtesting_figure()
pio.show(backtesting_figure)
```
Allocate capital.
```
optimizer.allocate_capital(54368)
optimizer.generate_capital_allocation_chart()
print(instanceRPO.capital_allocation)
```
# References
* Feng, Y., & Palomar, D. P. (2016). A signal processing perspective on financial engineering. Foundations and Trends® in Signal Processing, 9(1–2), 1-231.
* Yin, C., Perchet, R., & Soupé, F. (2021). A practical guide to robust portfolio optimization. Quantitative Finance, 21(6), 911-928.
