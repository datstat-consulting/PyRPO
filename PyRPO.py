"""Portfolio allocation, robust optimization and walk-forward evaluation."""

from numbers import Integral

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = ["PyRPO"]


def _frame(data):
    frame = pd.DataFrame(data, dtype=float).copy()
    if (len(frame) < 2 or not frame.shape[1] or not frame.columns.is_unique
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or not np.isfinite(frame.to_numpy()).all()):
        raise ValueError("Provide at least two finite, chronologically ordered rows with unique labels.")
    return frame


class PyRPO:
    """Optimize portfolios from prices or an explicitly supplied return history."""

    def __init__(self, csv_file=None, *, returns=None, periods_per_year=252, shrinkage=0.0):
        if (csv_file is None) == (returns is None):
            raise ValueError("Supply one of csv_file (or a price DataFrame) and returns.")
        if not isinstance(periods_per_year, Integral) or periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer.")
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0, 1].")
        self.periods_per_year = periods_per_year
        self.shrinkage = shrinkage
        self.data = None
        if returns is None:
            price_data = csv_file if isinstance(csv_file, pd.DataFrame) else pd.read_csv(
                csv_file, index_col=0, parse_dates=True
            )
            self.data = _frame(price_data)
            if (self.data <= 0).any().any():
                raise ValueError("Prices must be strictly positive.")
            returns = self.data.pct_change(fill_method=None).iloc[1:]
        self.returns = _frame(returns)
        self.asset_names = self.returns.columns
        self.n = len(self.asset_names)
        self._set_estimates(self.returns)
        self.optimal_weights = self.robust_type = self.delta = self.gamma = None
        self.portfolio_returns = self.equal_returns = None

    def _set_estimates(self, returns):
        self._estimation_returns = returns
        self.expected_returns = returns.mean()
        covariance = returns.cov()
        diagonal = np.diag(np.diag(covariance))
        self.covariance_matrix = covariance * (1 - self.shrinkage) + diagonal * self.shrinkage
        self.optimal_weights = self.portfolio_returns = self.equal_returns = None

    def train_test_split(self, test_size=0.2):
        """Chronologically split the history and fit moments on the training rows."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must lie strictly between zero and one.")
        boundary = len(self.returns) - int(np.ceil(len(self.returns) * test_size))
        if boundary < 2:
            raise ValueError("At least two training returns are required.")
        train, test = self.returns.iloc[:boundary], self.returns.iloc[boundary:]
        self._set_estimates(train)
        return train.copy(), test.copy()

    def sharpe_ratio(self, daily_returns, risk_free_rate=0.02, time_period=None):
        """Annualize the sample Sharpe ratio with an annual effective risk-free rate."""
        periods = self.periods_per_year if time_period is None else time_period
        excess = pd.Series(daily_returns) - self._per_period_rate(risk_free_rate, periods)
        volatility = excess.std()
        return excess.mean() / volatility * np.sqrt(periods) if volatility > 0 else np.nan

    @staticmethod
    def _per_period_rate(rate, periods):
        if not np.isfinite(rate) or rate <= -1 or not np.isfinite(periods) or periods <= 0:
            raise ValueError("Require risk-free rate > -1 and a positive number of periods.")
        return np.expm1(np.log1p(rate) / periods)

    def _risk_factor(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        return np.sqrt(eigenvalues.clip(0))[:, None] * eigenvectors.T

    def _vector(self, value):
        if isinstance(value, pd.Series):
            value = value.reindex(self.asset_names)
        values = np.broadcast_to(np.asarray(value, dtype=float), (self.n,))
        if not np.isfinite(values).all():
            raise ValueError("Provide finite values for each asset.")
        return values

    def _bounds(self, bounds):
        lower, upper = map(self._vector, bounds)
        if np.any(lower > upper) or lower.sum() > 1 + 1e-10 or upper.sum() < 1 - 1e-10:
            raise ValueError("Weight bounds must admit a fully invested portfolio.")
        return lower, upper

    def _robust_return(self, weights, rho, robust_type):
        if robust_type not in {"ellipsoidal", "box"}:
            raise ValueError("robust_type must be 'ellipsoidal' or 'box'.")
        nominal = self.expected_returns.to_numpy() @ weights
        if rho == 0:
            return nominal
        errors = np.sqrt(np.diag(self.covariance_matrix).clip(0) / len(self._estimation_returns))
        exposure = cp.multiply(errors, weights)
        penalty = cp.norm(exposure, 2) if robust_type == "ellipsoidal" else cp.norm1(exposure)
        return nominal - rho * penalty

    @staticmethod
    def _solve(objective, constraints, solver):
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"Portfolio optimization failed with status {problem.status}.")

    def _save(self, weights, robust_type, rho, gamma=None):
        if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1, atol=1e-5):
            raise ValueError("Solver returned invalid portfolio weights.")
        self.optimal_weights = np.array(weights, copy=True)
        self.robust_type, self.delta, self.gamma = robust_type, rho, gamma
        self.portfolio_returns = self.equal_returns = None
        return self.optimal_weights.copy()

    def optimize(self, objective="mean_variance", *, bounds=(0, 1), gamma=5.0,
                 rho=0.0, robust_type="ellipsoidal", target_return=None,
                 previous_weights=None, max_turnover=None, confidence=0.95, solver=None):
        """Solve a fully invested allocation with asset bounds and optional turnover cap.

        Objectives are minimum_variance, mean_variance, maximum_return and cvar.
        Return targets use the input observation period. Turnover is half the L1
        distance between the pre-trade portfolio and the new weights.
        """
        if not all(np.isfinite(x) and x >= 0 for x in (gamma, rho)):
            raise ValueError("gamma and rho must be finite and nonnegative.")
        lower, upper = self._bounds(bounds)
        weights = cp.Variable(self.n)
        constraints = [cp.sum(weights) == 1, weights >= lower, weights <= upper]
        robust_return = self._robust_return(weights, rho, robust_type)
        if target_return is not None:
            if not np.isfinite(target_return):
                raise ValueError("target_return must be finite.")
            constraints.append(robust_return >= target_return)
        if max_turnover is not None:
            if previous_weights is None or not np.isfinite(max_turnover) or max_turnover < 0:
                raise ValueError("A nonnegative turnover cap requires previous_weights.")
            previous = self._vector(previous_weights)
            if not np.isclose(previous.sum(), 1):
                raise ValueError("previous_weights must sum to one.")
            constraints.append(cp.norm1(weights - previous) <= 2 * max_turnover)
        variance = cp.sum_squares(self._risk_factor() @ weights)
        if objective == "minimum_variance":
            criterion = cp.Minimize(variance)
        elif objective == "mean_variance":
            criterion = cp.Maximize(robust_return - gamma * variance)
        elif objective == "maximum_return":
            criterion = cp.Maximize(robust_return)
        elif objective == "cvar":
            if not 0 < confidence < 1:
                raise ValueError("confidence must lie in (0, 1).")
            threshold = cp.Variable()
            losses = -self._estimation_returns.to_numpy() @ weights
            criterion = cp.Minimize(threshold + cp.sum(cp.pos(losses - threshold)) /
                                    (len(self._estimation_returns) * (1 - confidence)))
        else:
            raise ValueError(f"Unknown objective {objective!r}.")
        self._solve(criterion, constraints, solver)
        return self._save(weights.value, robust_type, rho, gamma)

    def solve_robust_mean_variance(self, gamma=5.0, rho=0.5, robust_type="ellipsoidal", **kwargs):
        return self.optimize("mean_variance", gamma=gamma, rho=rho,
                             robust_type=robust_type, **kwargs)

    def solve_robust_sharpe(self, risk_free_rate=0.02, rho=0.5, robust_type="ellipsoidal",
                            *, bounds=(0, 1), solver=None):
        """Maximize robust Sharpe over portfolios with positive robust excess return."""
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("rho must be finite and nonnegative.")
        lower, upper = self._bounds(bounds)
        rate = self._per_period_rate(risk_free_rate, self.periods_per_year)
        direction = cp.Variable(self.n)
        scale = cp.sum(direction)
        excess = self._robust_return(direction, rho, robust_type) - rate * scale
        risk_factor = self._risk_factor()
        constraints = [direction >= lower * scale, direction <= upper * scale, excess >= 1]
        self._solve(cp.Minimize(cp.norm(risk_factor @ direction, 2)), constraints, solver)
        if scale.value <= 0 or np.linalg.norm(risk_factor @ direction.value) <= 1e-10:
            raise ValueError("Sharpe maximization requires positive risk and a positive rescaling.")
        return self._save(direction.value / scale.value, robust_type, rho)

    def efficient_frontier(self, targets, **kwargs):
        """Return allocations and estimated moments at the requested return targets."""
        prior = (self.optimal_weights, self.robust_type, self.delta, self.gamma,
                 self.portfolio_returns, self.equal_returns)
        rows = []
        try:
            for target in targets:
                weights = self.optimize("minimum_variance", target_return=target, **kwargs)
                rows.append((weights @ self.expected_returns,
                             np.linalg.norm(self._risk_factor() @ weights), *weights))
        finally:
            (self.optimal_weights, self.robust_type, self.delta, self.gamma,
             self.portfolio_returns, self.equal_returns) = prior
        columns = pd.MultiIndex.from_tuples(
            [("moment", "expected_return"), ("moment", "volatility")]
            + [("weight", asset) for asset in self.asset_names]
        )
        return pd.DataFrame(rows, columns=columns)

    def backtest(self, test_returns):
        """Backtest fixed target weights rebalanced each period without costs."""
        if self.optimal_weights is None:
            raise RuntimeError("Optimize portfolio weights first.")
        test_returns = pd.DataFrame(test_returns).loc[:, self.asset_names]
        if not np.isfinite(test_returns.to_numpy()).all():
            raise ValueError("Test returns must be finite.")
        portfolio = test_returns @ self.optimal_weights
        equal = test_returns.mean(axis=1)
        self.portfolio_returns = (1 + portfolio).cumprod()
        self.equal_returns = (1 + equal).cumprod()
        return portfolio, self.portfolio_returns, equal, self.equal_returns

    def walk_forward(self, lookback, *, rebalance_every=1, transaction_cost=0.0,
                     objective="mean_variance", **kwargs):
        """Refit each rebalance on preceding returns, with drift and proportional costs."""
        if (not isinstance(lookback, Integral) or not isinstance(rebalance_every, Integral)
                or not 2 <= lookback < len(self.returns) or rebalance_every < 1):
            raise ValueError("Provide a valid lookback and positive rebalance interval.")
        if not np.isfinite(transaction_cost) or not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must lie in [0, 1).")
        if np.any(self.returns.to_numpy() < -1):
            raise ValueError("Simple returns cannot be below -100%.")
        weights = np.full(self.n, 1 / self.n)
        rows, positions = [], []
        for index in range(lookback, len(self.returns)):
            turnover = 0.
            if (index - lookback) % rebalance_every == 0:
                fit = PyRPO(returns=self.returns.iloc[index - lookback:index],
                            periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
                target = fit.optimize(objective, previous_weights=weights, **kwargs)
                turnover = np.abs(target - weights).sum() / 2
                weights = target
            values = self.returns.iloc[index].to_numpy()
            positions.append(weights.copy())
            growth = 1 + weights @ values
            after_cost = growth * (1 - transaction_cost * turnover)
            if growth <= 0 or after_cost <= 0:
                raise ValueError("Portfolio wealth was exhausted.")
            weights = weights * (1 + values) / growth
            rows.append((after_cost - 1, turnover))
        result = pd.DataFrame(rows, index=self.returns.index[lookback:],
                              columns=["return", "turnover"])
        result["wealth"] = (1 + result["return"]).cumprod()
        return result, pd.DataFrame(positions, index=result.index, columns=self.asset_names)

    def plot_weights(self):
        if self.optimal_weights is None:
            raise RuntimeError("Optimize portfolio weights first.")
        return pd.Series(self.optimal_weights, index=self.asset_names).plot.bar(ylabel="Weight")

    def plot_backtest(self):
        if self.portfolio_returns is None:
            raise RuntimeError("Run backtest first.")
        return pd.DataFrame({"Portfolio": self.portfolio_returns,
                             "Equal weight": self.equal_returns}).plot(ylabel="Wealth")

    def plot_capital_allocation(self, initial_capital=1_000_000):
        if self.optimal_weights is None:
            raise RuntimeError("Optimize portfolio weights first.")
        allocation = pd.Series(initial_capital * self.optimal_weights, index=self.asset_names)
        return allocation.plot.bar(ylabel="Capital allocation")
