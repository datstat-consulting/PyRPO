import numpy as np
import pandas as pd

from PyRPO import PyRPO


def test_training_split_and_robust_allocations():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(0.001, 0.005, (100, 3)), columns=list("ABC"))
    frame.iloc[-20:] -= 0.004
    model = PyRPO(returns=frame, shrinkage=0.1)
    train, test = model.train_test_split(0.2)
    np.testing.assert_allclose(model.expected_returns, train.mean())
    for objective in ("minimum_variance", "mean_variance", "maximum_return", "cvar"):
        weights = model.optimize(objective, rho=0.5, bounds=(0, 0.6))
        assert np.isclose(weights.sum(), 1)
        assert weights.min() >= -1e-5 and weights.max() <= 0.60001
    weights = model.solve_robust_sharpe(rho=0.1, risk_free_rate=0)
    original, *rest = model.backtest(test)
    reordered, *_ = model.backtest(test[["C", "A", "B"]])
    np.testing.assert_allclose(original, reordered)
    frontier = model.efficient_frontier([0., 0.0003])
    assert frontier.shape == (2, 5)
    np.testing.assert_array_equal(model.optimal_weights, weights)


def test_walk_forward_ignores_returns_at_rebalance():
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(0, 0.01, (50, 2)))
    original, weights = PyRPO(returns=returns).walk_forward(
        20, rebalance_every=5, transaction_cost=0.002, max_turnover=0.1
    )
    returns.iloc[20] += [0.1, -0.1]
    changed, changed_weights = PyRPO(returns=returns).walk_forward(
        20, rebalance_every=5, transaction_cost=0.002, max_turnover=0.1
    )
    np.testing.assert_allclose(weights.iloc[0], changed_weights.iloc[0], atol=1e-9)
    assert original["return"].iloc[0] != changed["return"].iloc[0]
    assert original["turnover"].max() <= 0.10001
