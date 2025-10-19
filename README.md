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
from PyRPO import PyRPO
import numpy as np
import pandas as pd

rpo = PyRPO('PortfolioData.csv')
train_data, test_data = rpo.train_test_split(test_size=0.2)
```
Set the risk aversion parameter (gamma) and solve the RPO problem.
```
# Choose ONE of these:

# A) Robust mean–variance (ellipsoidal mean uncertainty)
w_star = rpo.solve_robust_mean_variance(
    gamma=0.5,         # risk aversion on variance
    rho=0.4,           # robustness strength (penalty)
    robust_type="ellipsoidal"  # or "box"
)
print("Optimal weights (RMV):", w_star)

# OR B) Robust Sharpe ratio (box mean uncertainty)
# w_star = rpo.solve_robust_sharpe(
#     risk_free_rate=0.05,
#     rho=0.5,
#     robust_type="box"         # or "ellipsoidal"
# )
# print("Optimal weights (Robust Sharpe):", w_star)
```
Backtest.
```
pd_port, pc_port, pd_eq, pc_eq = rpo.backtest(test_returns=test_data)  # test_data are returns (already handled)
print("OOS Sharpe (robust):", rpo.sharpe_ratio(pd_port))
print("OOS Sharpe (equal) :", rpo.sharpe_ratio(pd_eq))
```
Plot the optimal weights and sensitivity analyses.
```
rpo.plot_weights()
rpo.plot_backtest()
```
Allocate capital.
```
rpo.plot_capital_allocation(initial_capital=54368)
```
# References
* Feng, Y., & Palomar, D. P. (2016). A signal processing perspective on financial engineering. Foundations and Trends® in Signal Processing, 9(1–2), 1-231.
* Georgantas, A., Doumpos, M., & Zopounidis, C. (2021). Robust optimization approaches for portfolio selection: a comparative analysis. Annals of Operations Research, 1-17.
* Yin, C., Perchet, R., & Soupé, F. (2021). A practical guide to robust portfolio optimization. Quantitative Finance, 21(6), 911-928.
