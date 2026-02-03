"""
Knowledge graph components for database schema representation.

Includes schema ontology and graph-based schema management.
"""

from .schema_ontology import DatabaseSchemaKG, TableInfo, JoinInfo

__all__ = [
    "DatabaseSchemaKG",
    "TableInfo",
    "JoinInfo"
]