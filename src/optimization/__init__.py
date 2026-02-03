"""
Optimization package for intelligent database query optimization.

This package contains the core optimization algorithms and strategies
used by the intelligent database optimizer system.
"""

from .query_optimizer import QueryOptimizer
from .cost_estimator import CostEstimator

__all__ = ['QueryOptimizer', 'CostEstimator']