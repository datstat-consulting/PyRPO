"""
Разработанный Адриелу Ванг от ДанСтат Консульти́рования
"""

import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
import plotly.express as px

class PyRPO:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        self.asset_names = self.data.columns
        self.returns = self.data.pct_change().dropna()
        self.expected_returns = self.returns.mean()
        self.covariance_matrix = self.returns.cov()
        self.n = len(self.expected_returns)
        self.optimal_weights = None
        self.uncertainty_factors = None
        self.sensitivity_weights = None
        self.capital_allocation = None
        self.optimal_portfolio_daily_returns = None
        self.optimal_portfolio_cumulative_returns = None
        self.equally_weighted_weights = None
        self.equally_weighted_daily_returns = None
        self.equally_weighted_cumulative_returns = None

    def objective_function(self, weights, gamma, uncertainty_radius):
        portfolio_risk = weights @ self.covariance_matrix @ weights
        expected_portfolio_returns = np.dot(weights, self.expected_returns)
        return expected_portfolio_returns + uncertainty_radius * np.sqrt(weights @ np.diag(np.diag(self.covariance_matrix)) @ weights) - gamma*portfolio_risk

    def constraints_sum_to_one(self, weights):
        return np.sum(weights) - 1

    def uncertainty_estimate(self, risk_free_rate):
        # Calculate daily excess returns
        excess_returns = self.returns - risk_free_rate/252

        # Calculate the Sharpe ratio for each asset
        sharpe_ratios = excess_returns.mean() / excess_returns.std()

        # Calculate the average Sharpe ratio across all assets
        average_sharpe_ratio = sharpe_ratios.mean()

        return average_sharpe_ratio/2

    def solve_rpo(self, gamma, uncertainty_radius = None, risk_free_rate = 0.02):
        if uncertainty_radius == None:
            # Method taken from
            # Yin, C., Perchet, R., & Soupé, F. (2021). A practical guide to robust portfolio optimization. Quantitative Finance, 21(6), 911-928.
            uncertainty_radius = self.uncertainty_estimate(risk_free_rate)

        num_assets = len(self.expected_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = ({'type': 'eq', 'fun': self.constraints_sum_to_one})
        bounds = [(0, 1) for _ in range(num_assets)]

        result = minimize(self.objective_function, initial_weights, args=(gamma, uncertainty_radius),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        self.optimal_weights = result.x
        return result.x

    # uncertainty_bounds_factor_range = np.linspace(x, y, z)
    def sensitivity_analysis(self, uncertainty_radius_range, gamma):
        uncertainty_radii = uncertainty_radius_range
        # Initialize the list of sensitivity weights
        self.sensitivity_weights = []

        # Define the uncertainty_constraint function
        def uncertainty_constraint(weights, radius):
            return radius - np.linalg.norm((self.returns - self.expected_returns).values @ weights)

        # Loop over the uncertainty_radius_range
        for radius in uncertainty_radius_range:
            # Define constraints
            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "ineq", "fun": lambda w: w},
                {"type": "ineq", "fun": lambda w: uncertainty_constraint(w, radius)},
            )
            objective = lambda w: w @ self.expected_returns + radius * np.sqrt(w @ np.diag(np.diag(self.covariance_matrix)) @ w) - gamma * (w.T @ self.covariance_matrix @ w)
            result = minimize(objective, self.optimal_weights, constraints=constraints)

            # Check if the optimization problem is solvable
            if not result.success:
                raise ValueError("The optimization problem could not be solved.")

            self.sensitivity_weights.append(result.x)

    # Run backtesting
    def backtesting(self):
        # Calculate the daily returns of the optimal portfolio
        self.optimal_portfolio_daily_returns = self.returns.dot(self.optimal_weights)

        # Calculate the cumulative returns of the optimal portfolio
        self.optimal_portfolio_cumulative_returns = (self.optimal_portfolio_daily_returns + 1).cumprod()

        # Calculate the equally-weighted portfolio daily returns
        self.equally_weighted_weights = np.repeat(1 / self.n, self.n)
        self.equally_weighted_daily_returns = self.returns.dot(self.equally_weighted_weights)

        # Calculate the cumulative returns of the equally-weighted portfolio
        self.equally_weighted_cumulative_returns = (self.equally_weighted_daily_returns + 1).cumprod()

    # Plot the optimal weights using matplotlib
    def plot_optimal_weights(self, risk_free_rate, uncertainty_radius = None, x=10, y=6):
        if uncertainty_radius == None:
            uncertainty_radius = self.uncertainty_estimate(risk_free_rate)

        fig, ax = plt.subplots(figsize=(x, y))
        ax.errorbar(self.expected_returns.index, self.optimal_weights, yerr=[uncertainty_radius] * self.n, fmt='o', capsize=5)
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weights')
        ax.set_title('Optimal Portfolio Weights with Uncertainty')
        plt.show()

    # Plot the backtesting results using matplotlib
    def plot_backtesting(self, x = 10, y = 6):
        plt.figure(figsize=(x, y))
        plt.plot(self.optimal_portfolio_cumulative_returns, label='Optimal Portfolio')
        plt.plot(self.equally_weighted_cumulative_returns, label='Equally-Weighted Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Backtesting')
        plt.legend()
        plt.show()

    # Plot the sensitivity analysis results using matplotlib
    def plot_sensitivity_analysis(self, uncertainty_radii, x=12, y=8):
        plt.figure(figsize=(x, y))
        for i in range(self.n):
            plt.plot(uncertainty_radii, [w[i] for w in self.sensitivity_weights], label=self.asset_names[i])
        plt.xlabel('Uncertainty Radius')
        plt.ylabel('Optimal Weights')
        plt.legend()
        plt.title('Sensitivity Analysis of Portfolio Weights')
        plt.show()

    # Generate the optimal weights figure using Plotly
    def generate_optimal_weights_figure(self, risk_free_rate, uncertainty_radius = None):
        if uncertainty_radius == None:
            uncertainty_radius = self.uncertainty_estimate(risk_free_rate)

        trace = go.Scatter(
            x=self.expected_returns.index,
            y=self.optimal_weights,
            mode='markers',
            error_y=dict(
                type='data',
                array=[uncertainty_radius] * self.n,
                visible=True
            )
        )
        layout = go.Layout(
            title='Optimal Portfolio Weights with Uncertainty',
            xaxis=dict(title='Assets'),
            yaxis=dict(title='Weights')
        )
        return go.Figure(data=[trace], layout=layout)

    # Generate the backtesting figure using Plotly
    def generate_backtesting_figure(self):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=self.returns.index, y=self.optimal_portfolio_cumulative_returns, mode='lines', name='Optimal Portfolio'))
        figure.add_trace(go.Scatter(x=self.returns.index, y=self.equally_weighted_cumulative_returns, mode='lines', name='Equally-Weighted Portfolio'))
        return figure

    # Generate the sensitivity analysis figure using Plotly
    def generate_sensitivity_analysis_figure(self, uncertainty_radii):
        fig = go.Figure()
        for i in range(self.n):
            fig.add_trace(go.Scatter(x=uncertainty_radii, y=[w[i] for w in self.sensitivity_weights],
                                    mode='lines', name=self.asset_names[i]))
        fig.update_layout(title='Sensitivity Analysis of Portfolio Weights',
                        xaxis_title='Uncertainty Radius',
                        yaxis_title='Optimal Weights')
        return fig

    # Allocate capital among assets
    def allocate_capital(self, initial_capital):
        if self.optimal_weights is None:
            raise ValueError("Optimal weights have not been calculated. Call solve_rpo() method first.")
        
        capital_allocation = np.multiply(initial_capital, self.optimal_weights)
        #self.capital_allocation = pd.DataFrame(capital_allocation, index=self.asset_names, columns = ["Capital Allocation"])
        self.capital_allocation = capital_allocation.transpose()
        fig, ax = plt.subplots()
        ax.pie(capital_allocation, labels=self.asset_names)
        
    def generate_capital_allocation_chart(self):
        fig = px.pie(values=self.capital_allocation, names=self.asset_names)
        fig.show()
