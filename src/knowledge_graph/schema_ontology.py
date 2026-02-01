"""
Database schema knowledge graph representation.
Supports both SQLite and PostgreSQL databases.
"""
import sqlite3
import psycopg2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logging import logger

@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[str]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[Dict[str, str]] = None  # column -> referenced_table.column
    row_count: int = 0

@dataclass
class JoinInfo:
    """Information about a join between two tables."""
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    join_type: str = "INNER"  # e.g., INNER, LEFT, RIGHT

class DatabaseMetadataExtractor(ABC):
    """Abstract base class for database metadata extraction."""
    
    @abstractmethod
    def get_table_names(self, cursor) -> List[str]:
        """Get all table names from the database."""
        pass
    
    @abstractmethod
    def get_table_columns(self, cursor, table_name: str) -> List[str]:
        """Get column names for a specific table."""
        pass
    
    @abstractmethod
    def get_primary_keys(self, cursor, table_name: str) -> List[str]:
        """Get primary key columns for a specific table."""
        pass
    
    @abstractmethod
    def get_foreign_keys(self, cursor, table_name: str) -> Dict[str, str]:
        """Get foreign key mappings for a specific table."""
        pass
    
    def get_row_count(self, cursor, table_name: str) -> int:
        """Get row count for a specific table (common for both databases)."""
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        return cursor.fetchone()[0]

class SQLiteMetadataExtractor(DatabaseMetadataExtractor):
    """SQLite-specific metadata extraction."""
    
    def get_table_names(self, cursor) -> List[str]:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_columns(self, cursor, table_name: str) -> List[str]:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        return [col[1] for col in columns_info]  # col[1] is column name
    
    def get_primary_keys(self, cursor, table_name: str) -> List[str]:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        return [col[1] for col in columns_info if col[5] == 1]  # col[5] is pk flag
    
    def get_foreign_keys(self, cursor, table_name: str) -> Dict[str, str]:
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fk_info = cursor.fetchall()
        
        foreign_keys = {}
        for fk in fk_info:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            from_col = fk[3]  # from column
            to_table = fk[2]  # referenced table
            to_col = fk[4]    # referenced column
            foreign_keys[from_col] = f"{to_table}.{to_col}"
        
        return foreign_keys

class PostgreSQLMetadataExtractor(DatabaseMetadataExtractor):
    """PostgreSQL-specific metadata extraction."""
    
    def get_table_names(self, cursor) -> List[str]:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_columns(self, cursor, table_name: str) -> List[str]:
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position;
        """, (table_name,))
        return [row[0] for row in cursor.fetchall()]
    
    def get_primary_keys(self, cursor, table_name: str) -> List[str]:
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY';
        """, (table_name,))
        return [row[0] for row in cursor.fetchall()]
    
    def get_foreign_keys(self, cursor, table_name: str) -> Dict[str, str]:
        cursor.execute("""
            SELECT 
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
        """, (table_name,))
        
        fk_info = cursor.fetchall()
        foreign_keys = {}
        for fk in fk_info:
            from_col = fk[0]
            to_table = fk[1]
            to_col = fk[2]
            foreign_keys[from_col] = f"{to_table}.{to_col}"
        
        return foreign_keys

class DatabaseSchemaKG:
    """Knowledge graph representation of a database schema.
    Supports both SQLite and PostgreSQL databases.
    """
    
    def __init__(self, db_type: str = "sqlite"):
        self.db_type = db_type.lower()
        self.tables: Dict[str, TableInfo] = {}
        self.relationships: List[JoinInfo] = []
        self._extractor = self._get_metadata_extractor()
    
    def _get_metadata_extractor(self) -> DatabaseMetadataExtractor:
        """Factory method to get the appropriate metadata extractor."""
        if self.db_type == "sqlite":
            return SQLiteMetadataExtractor()
        elif self.db_type == "postgresql":
            return PostgreSQLMetadataExtractor()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def add_table(self, table_info: TableInfo):
        """Add a table to the schema knowledge graph."""
        self.tables[table_info.name] = table_info
        logger.info(f"Added table: {table_info.name}")
    
    def add_relationship(self, join_info: JoinInfo):
        """Add a relationship (join) between two tables."""
        self.relationships.append(join_info)
        logger.info(f"Added relationship: {join_info.left_table} {join_info.join_type} JOIN {join_info.right_table} ON {join_info.left_column} = {join_info.right_column}")
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Retrieve information about a specific table."""
        return self.tables.get(table_name)
    
    def get_related_tables(self, table_name: str) -> List[str]:
        """Get list of tables that can be joined with the given table."""
        related = []
        
        for rel in self.relationships:
            if rel.left_table == table_name and rel.right_table not in related:
                related.append(rel.right_table)
            elif rel.right_table == table_name and rel.left_table not in related:
                related.append(rel.left_table)
                
        return related
    
    def get_join_info(self, table1: str, table2: str) -> Optional[JoinInfo]:
        """Get join information between two tables if it exists."""
        for rel in self.relationships:
            if (rel.left_table == table1 and rel.right_table == table2) or \
               (rel.left_table == table2 and rel.right_table == table1):
                return rel
        
        return None
    
    def suggest_join_order(self, tables: List[str]) -> List[str]:
        """Suggest optimal join order based on table sizes."""
        if len(tables) <= 1:
            return tables
        
        # Simple heuristic: start with smallest table
        table_sizes = {
            table_name: self.get_table_info(table_name).row_count 
            if self.get_table_info(table_name) else 0
            for table_name in tables
        }
        
        # Sort tables by size (smallest first)
        sorted_tables = sorted(tables, key=lambda t: table_sizes.get(t, 0))
        logger.info(f"Suggested join order based on table sizes: {sorted_tables}")
        return sorted_tables
    
    def build_from_database(self, db_connection):
        """Build knowledge graph from database metadata."""
        cursor = db_connection.cursor()
        
        try:
            # Get all table names
            table_names = self._extractor.get_table_names(cursor)
            logger.info(f"Found {len(table_names)} tables: {table_names}")
            
            # Process each table
            for table_name in table_names:
                table_info = self._extract_table_info(cursor, table_name)
                self.add_table(table_info)
                self._add_relationships_from_foreign_keys(table_info)
            
            logger.info(f"Knowledge graph built successfully with {len(self.tables)} tables and {len(self.relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error building schema from {self.db_type} database: {e}")
            raise
        finally:
            cursor.close()
    
    def _extract_table_info(self, cursor, table_name: str) -> TableInfo:
        """Extract complete table information using the metadata extractor."""
        columns = self._extractor.get_table_columns(cursor, table_name)
        primary_keys = self._extractor.get_primary_keys(cursor, table_name)
        foreign_keys = self._extractor.get_foreign_keys(cursor, table_name)
        row_count = self._extractor.get_row_count(cursor, table_name)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_key=primary_keys if primary_keys else None,
            foreign_keys=foreign_keys if foreign_keys else None,
            row_count=row_count
        )
    
    def _add_relationships_from_foreign_keys(self, table_info: TableInfo):
        """Add relationships based on foreign key constraints."""
        if not table_info.foreign_keys:
            return
        
        for from_col, referenced in table_info.foreign_keys.items():
            if '.' in referenced:
                ref_table, ref_col = referenced.split('.')
                join_info = JoinInfo(
                    left_table=table_info.name,
                    right_table=ref_table,
                    left_column=from_col,
                    right_column=ref_col
                )
                self.add_relationship(join_info)
    
    def print_summary(self):
        """Print a summary of the knowledge graph."""
        print("\n=== Database Schema Knowledge Graph ===")
        
        print(f"\nTables ({len(self.tables)}):")
        for table_name, table_info in self.tables.items():
            print(f"  - {table_name}: {len(table_info.columns)} columns, {table_info.row_count} rows")
            print(f"    Columns: {', '.join(table_info.columns)}")
            if table_info.primary_key:
                print(f"    Primary Keys: {', '.join(table_info.primary_key)}")
            if table_info.foreign_keys:
                print(f"    Foreign Keys: {table_info.foreign_keys}")
        
        print(f"\nRelationships ({len(self.relationships)}):")
        for rel in self.relationships:
            print(f"  - {rel.left_table}.{rel.left_column} -> {rel.right_table}.{rel.right_column}")