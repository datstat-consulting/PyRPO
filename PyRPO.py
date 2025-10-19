"""
Разработанный Адриелу Ванг от ДанСтат Консульти́рования

Robust Portfolio Optimization (RPO)
Implemented following robust mean–variance and Sharpe ratio formulations
from the literature.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import plotly.express as px


class PyRPO:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        self.asset_names = self.data.columns
        self.returns = self.data.pct_change().dropna()
        self.expected_returns = self.returns.mean()
        self.covariance_matrix = self.returns.cov()
        self.n = len(self.expected_returns)

        # outputs
        self.optimal_weights = None
        self.robust_type = None
        self.delta = None
        self.gamma = None
        self.portfolio_returns = None
        self.equal_returns = None

    # ----------------------------------------------------------
    # Basic utilities
    # ----------------------------------------------------------

    def train_test_split(self, test_size=0.2):
        n_samples = len(self.returns)
        test_samples = int(n_samples * test_size)
        train = self.returns.iloc[:-test_samples]
        test = self.returns.iloc[-test_samples:]
        return train, test

    def sharpe_ratio(self, daily_returns, risk_free_rate=0.02, time_period=252):
        excess = daily_returns - risk_free_rate / time_period
        return excess.mean() / excess.std() * np.sqrt(time_period)

    # ----------------------------------------------------------
    # Robust optimization solvers
    # ----------------------------------------------------------

    def solve_robust_mean_variance(self, gamma=5.0, rho=0.5, robust_type="ellipsoidal"):
        """
        Solve robust mean–variance optimization:
        max_w min_{μ∈Uμ} wᵀμ - γ wᵀΣw
        s.t. 1ᵀw = 1, w >= 0
        where:
          - Uμ: mean uncertainty set (box or ellipsoidal)
        """
        mu_hat = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        n = len(mu_hat)
        w = cp.Variable(n)
        gamma = float(gamma)
        rho = float(rho)

        if robust_type == "box":
            # Box mean uncertainty ⇒ |w|ᵀδ penalty
            obj = cp.Maximize(mu_hat @ w - rho * cp.norm1(w) - gamma * cp.quad_form(w, Sigma))
        elif robust_type == "ellipsoidal":
            # Ellipsoidal mean uncertainty ⇒ ρ‖w‖₂ penalty
            obj = cp.Maximize(mu_hat @ w - rho * cp.norm(w, 2) - gamma * cp.quad_form(w, Sigma))
        else:
            raise ValueError("robust_type must be 'box' or 'ellipsoidal'")

        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            raise ValueError("Optimization failed; check inputs or solver.")

        self.optimal_weights = w.value
        self.robust_type = robust_type
        self.gamma = gamma
        self.delta = rho
        return self.optimal_weights

    # ----------------------------------------------------------
    # Robust Sharpe ratio optimization
    # ----------------------------------------------------------

    def solve_robust_sharpe(self, risk_free_rate=0.02, rho=0.5, robust_type="ellipsoidal"):
        """
        Robust Sharpe ratio maximization (min variance s.t. robust excess return ≥ 1):
        min_w wᵀΣw
        s.t. min_{μ∈Uμ} wᵀ(μ - rf*1) ≥ 1, 1ᵀw = 1, w ≥ 0
        """
        mu_hat = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        rf = risk_free_rate / 252
        n = len(mu_hat)
        w = cp.Variable(n)
        rho = float(rho)

        if robust_type == "box":
            # robust excess return constraint
            excess_constraint = (mu_hat - rf) @ w - rho * cp.norm1(w) >= 1
        elif robust_type == "ellipsoidal":
            excess_constraint = (mu_hat - rf) @ w - rho * cp.norm(w, 2) >= 1
        else:
            raise ValueError("robust_type must be 'box' or 'ellipsoidal'")

        obj = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [excess_constraint, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            raise ValueError("Optimization failed; check inputs or solver.")

        self.optimal_weights = w.value
        self.robust_type = robust_type
        self.delta = rho
        return self.optimal_weights

    # ----------------------------------------------------------
    # Backtesting
    # ----------------------------------------------------------

    def backtest(self, test_returns):
        if self.optimal_weights is None:
            raise ValueError("Call solve_robust_mean_variance() or solve_robust_sharpe() first.")

        # portfolio daily returns
        port_daily = test_returns @ self.optimal_weights
        port_cum = (1 + port_daily).cumprod()

        # equally weighted benchmark
        w_eq = np.ones(test_returns.shape[1]) / test_returns.shape[1]
        eq_daily = test_returns @ w_eq
        eq_cum = (1 + eq_daily).cumprod()

        self.portfolio_returns = port_cum
        self.equal_returns = eq_cum
        return port_daily, port_cum, eq_daily, eq_cum

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------

    def plot_weights(self):
        if self.optimal_weights is None:
            raise ValueError("No optimal weights computed.")
        plt.figure(figsize=(10, 6))
        plt.bar(self.asset_names, self.optimal_weights)
        plt.title(f"Optimal Weights ({self.robust_type} robust model)")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)
        plt.show()

    def plot_backtest(self):
        if self.portfolio_returns is None or self.equal_returns is None:
            raise ValueError("Run backtest() first.")
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_returns, label="Robust Portfolio")
        plt.plot(self.equal_returns, label="Equally Weighted")
        plt.title("Backtest: Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()

    # ----------------------------------------------------------
    # Capital allocation visualization
    # ----------------------------------------------------------

    def plot_capital_allocation(self, initial_capital=1_000_000):
        if self.optimal_weights is None:
            raise ValueError("No optimal weights available.")
        allocation = initial_capital * self.optimal_weights
        fig = px.pie(values=allocation, names=self.asset_names,
                     title=f"Capital Allocation ({self.robust_type})")
        fig.show()
