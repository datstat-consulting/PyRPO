"""
PyRPO: A Python package for Robust Portfolio Optimization.

This package provides an implementation of Robust Portfolio Optimization
based on the ellipsoid uncertainty set, allowing users to find optimal
portfolio weights considering estimation errors in expected returns and
covariance matrix.
"""

from .PyRPO import PyRPO

__all__ = ['PyRPO']
