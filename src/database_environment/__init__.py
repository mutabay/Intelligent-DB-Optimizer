"""
Database environment components.

Includes database simulators and connection management.
"""

from .db_simulator import DatabaseSimulator, QueryResult

__all__ = [
    "DatabaseSimulator",
    "QueryResult"
]