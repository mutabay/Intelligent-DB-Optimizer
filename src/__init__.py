"""
Intelligent Database Optimizer Package

A multi-agent reinforcement learning system for database query optimization.
"""

__version__ = "1.0.0"
__author__ = "PhD Research Project"

# Main components
from .agents.dqn_agent import DQNAgent, MultiAgentDQN
from .agents.rl_environment import QueryOptimizationEnv
from .database_environment.db_simulator import DatabaseSimulator
from .knowledge_graph.schema_ontology import DatabaseSchemaKG
from .optimization.query_optimizer import QueryOptimizer, OptimizationStrategy
from .utils.logging import setup_logging as setup_logger

__all__ = [
    "DQNAgent",
    "MultiAgentDQN", 
    "QueryOptimizationEnv",
    "DatabaseSimulator",
    "DatabaseSchemaKG",
    "QueryOptimizer",
    "OptimizationStrategy",
    "setup_logger"
]