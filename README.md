# PyRPO

Markowitz mean–variance allocations can be sensitive to errors in estimated expected returns. PyRPO adds a penalty for uncertainty in those estimates and supports portfolio weight bounds and turnover limits. It takes historical closing prices or simple asset returns and includes separate workflows for fixed-weight and walk-forward evaluation.

## Robust mean–variance objective

For an ellipsoidal mean uncertainty set, the objective is

$$
\max_{\mathbf w}\quad
\mathbf w^\top\widehat{\boldsymbol\mu}
- \rho\lVert D\mathbf w\rVert_2
- \gamma\mathbf w^\top\widehat{\Sigma}\mathbf w,
\qquad \mathbf 1^\top\mathbf w = 1.
$$

Here $\mathbf w$ contains the asset weights, $\widehat{\boldsymbol\mu}$ and $\widehat{\Sigma}$ are the sample mean and estimated covariance, and $D$ is diagonal with each asset's estimated mean standard error. The uncertainty radius $\rho$ scales the worst-case reduction in expected return. Set `robust_type="box"` to use an L1 penalty on the same scaled exposures, or `rho=0` for nominal optimization. The risk-aversion parameter $\gamma$ scales the variance penalty. Optional covariance shrinkage moves the estimate toward its diagonal.

## Installation

~~~bash
python -m pip install git+https://github.com/datstat-consulting/PyRPO.git
~~~

From a checkout, run `python -m pip install .` or `python -m pip install '.[plots]'` to include Matplotlib. Python 3.10 or later is required. The optimization problems are solved with CVXPY.

## Historical prices and a holdout backtest

Store one price per asset and timestamp in a CSV with a date index in the first column. For example, `PortfolioData.csv` could contain the columns `date,XRP/USD,ETH/USD,ADA/USD`. Align prices to the same frequency and resolve missing values before fitting. The package computes simple returns from the positive closing prices. Pass a DataFrame through `returns=` if the input already consists of simple returns.

The example uses synthetic data so it can be run without an exchange connection. Replace the `PyRPO(returns=...)` line with `PyRPO("PortfolioData.csv", periods_per_year=365, shrinkage=0.1)` for daily cryptocurrency prices.

~~~python
import numpy as np
import pandas as pd
from PyRPO import PyRPO

rng = np.random.default_rng(42)
returns = pd.DataFrame(
    rng.normal(0.001, 0.01, (240, 3)),
    columns=["A", "B", "C"],
)
model = PyRPO(returns=returns, periods_per_year=252, shrinkage=0.1)
train, test = model.train_test_split(test_size=0.2)
weights = model.solve_robust_mean_variance(
    gamma=5, rho=0.5, robust_type="ellipsoidal", bounds=(0, 0.7)
)
daily, wealth, equal_daily, equal_wealth = model.backtest(test)
print(pd.Series(weights, index=train.columns))
print(model.sharpe_ratio(daily), model.sharpe_ratio(equal_daily))
~~~

The split estimates moments on the earlier rows. `backtest` applies each portfolio's constant target weights at every subsequent observation and reports compounded wealth from one unit of capital. This comparison excludes trading costs.

With the plotting extra installed, `model.plot_weights()` shows the allocation, `model.plot_backtest()` compares wealth against equal weighting, and `model.plot_capital_allocation(initial_capital=54368)` displays the amounts implied by the optimized weights.

## Closing prices from an exchange

For the cryptocurrency example, `PortfolioData.csv` can be built from daily candles. This example requests one page for each market and retains only dates inside the specified historical interval. Install CCXT separately and select an exchange with coverage for the requested pairs and dates.

~~~python
def make_price_table(exchange, symbols, start, stop):
    import pandas as pd

    first = exchange.parse8601(start)
    stop_ms = exchange.parse8601(stop)
    day_ms = 86_400_000
    if not 1 <= (stop_ms - first) / day_ms <= 100:
        raise ValueError("Use a date interval of at most 100 daily candles.")

    columns = []
    for symbol in symbols:
        rows = exchange.fetch_ohlcv(symbol, "1d", since=first, limit=100)
        closes = {
            row[0]: row[4] for row in rows
            if first <= row[0] and row[0] + day_ms <= stop_ms
        }
        timestamps = pd.to_datetime(list(closes), unit="ms", utc=True)
        columns.append(pd.Series(list(closes.values()), index=timestamps, name=symbol))
    return pd.concat(columns, axis=1).sort_index().rename_axis("date")
~~~

For longer intervals, fetch successive pages and reconcile overlapping timestamps before exporting. Inspect gaps in the aligned price table before using `pct_change`. The optimizer treats each return row as one observation period. [CCXT candlestick documentation](https://github.com/ccxt/ccxt/wiki/Manual#ohlcv-candlestick-charts)


## Other objectives and the efficient frontier

Call `optimize` with `"minimum_variance"`, `"maximum_return"`, or `"cvar"`. CVaR minimizes historical tail loss at the supplied `confidence` level. `solve_robust_sharpe` maximizes robust excess return per unit of volatility and requires a feasible portfolio with positive robust excess return and positive volatility. Its `risk_free_rate`, like the rate furnished to `sharpe_ratio`, is annual and effective. `periods_per_year` converts it to the observation frequency.

~~~python
frontier = model.efficient_frontier([0.0002, 0.0004], bounds=(0, 0.7))
~~~

Each row contains the estimated return, volatility, and weights for a feasible target. Return targets, means, and volatilities use the input observation period. Bounds can be scalars, arrays in asset order, or Series indexed by asset name.

## Walk-forward evaluation

~~~python
results, positions = model.walk_forward(
    lookback=90,
    rebalance_every=20,
    transaction_cost=0.001,
    bounds=(0, 0.7),
    rho=0.5,
    max_turnover=0.2,
)
~~~

Starting from equal weights, the model fits on the preceding 90 observations and applies the resulting position to the next return. Between rebalances, weights drift with asset prices. Turnover is half the sum of absolute weight changes, measured against the pre-trade portfolio. The transaction cost is proportional to that turnover. `results` contains per-period net returns, turnover, and wealth, and `positions` contains the weights used for each return.

## References

- Feng, Y., & Palomar, D. P. (2016). A signal processing perspective on financial engineering. *Foundations and Trends in Signal Processing*, 9(1–2), 1–231.
- Georgantas, A., Doumpos, M., & Zopounidis, C. (2021). Robust optimization approaches for portfolio selection: a comparative analysis. *Annals of Operations Research*, 1–17.
- Yin, C., Perchet, R., & Soupé, F. (2021). A practical guide to robust portfolio optimization. *Quantitative Finance*, 21(6), 911–928.
