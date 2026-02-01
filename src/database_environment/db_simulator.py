"""
Database simulation environment for query optimization experiments.
"""
import sqlite3
import psycopg2
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import config
from src.utils.logging import logger

@dataclass
class QueryResult:
    """Query execution result with performance metrics."""
    query: str
    execution_time: float
    rows_returned: int
    error: Optional[str] = None

class DatabaseSimulator:
    """
    Database simulator for testing query optimization strategies.
    Supports both SQLite and PostgreSQL.
    """
    
    def __init__(self, db_type: str = None, db_path: str = None):
        self.db_type = db_type or config.database.default_db_type
        self.db_path = db_path or config.database.sqlite_db_path
        self.connection = None
        self.query_history: List[QueryResult] = []
        
    def connect(self):
        """Establish database connection."""
        try:
            if self.db_type == "sqlite":
                self.connection = sqlite3.connect(self.db_path)
                logger.info(f"Connected to SQLite database: {self.db_path}")
            elif self.db_type == "postgresql":
                self.connection = psycopg2.connect(
                    host=config.database.postgres_host,
                    port=config.database.postgres_port,
                    database=config.database.postgres_db,
                    user=config.database.postgres_user,
                    password=config.database.postgres_password
                )
                self.connection.autocommit = True
                logger.info(f"Connected to PostgreSQL database: {config.database.postgres_db}")
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str) -> QueryResult:
        """
        Execute a query and return performance metrics.
        
        Args:
            query: SQL query to execute
            
        Returns:
            QueryResult with execution metrics
        """
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = self.connection.cursor()
        start_time = time.time()
        
        try:
            # Execute the query
            cursor.execute(query)
            
            # Handle different query types
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                rows = cursor.fetchall()
            else:
                rows = []
                # For non-SELECT queries, commit if needed
                if self.db_type == "sqlite":
                    self.connection.commit()
            
            execution_time = time.time() - start_time
            
            result = QueryResult(
                query=query.strip(),
                execution_time=execution_time,
                rows_returned=len(rows)
            )
            
            self.query_history.append(result)
            logger.info(f"Query executed in {execution_time:.3f}s, returned {len(rows)} rows")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = QueryResult(
                query=query.strip(),
                execution_time=execution_time,
                rows_returned=0,
                error=str(e)
            )
            
            self.query_history.append(result)
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            
            return result
        finally:
            cursor.close()
    
    def create_sample_tables(self):
        """Create sample tables for testing (works for both SQLite and PostgreSQL)."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = self.connection.cursor()
        
        # Drop tables if they exist (for clean testing)
        drop_commands = [
            "DROP TABLE IF EXISTS lineitem",
            "DROP TABLE IF EXISTS orders", 
            "DROP TABLE IF EXISTS customers"
        ]
        
        # Create tables (compatible with both databases)
        if self.db_type == "sqlite":
            schema_commands = [
                """CREATE TABLE customers (
                    c_custkey INTEGER PRIMARY KEY,
                    c_name TEXT NOT NULL,
                    c_address TEXT,
                    c_nationkey INTEGER,
                    c_phone TEXT,
                    c_acctbal REAL,
                    c_mktsegment TEXT
                )""",
                
                """CREATE TABLE orders (
                    o_orderkey INTEGER PRIMARY KEY,
                    o_custkey INTEGER,
                    o_orderstatus TEXT,
                    o_totalprice REAL,
                    o_orderdate TEXT,
                    o_orderpriority TEXT,
                    o_clerk TEXT,
                    o_shippriority INTEGER,
                    FOREIGN KEY (o_custkey) REFERENCES customers(c_custkey)
                )""",
                
                """CREATE TABLE lineitem (
                    l_orderkey INTEGER,
                    l_partkey INTEGER,
                    l_suppkey INTEGER,
                    l_linenumber INTEGER,
                    l_quantity REAL,
                    l_extendedprice REAL,
                    l_discount REAL,
                    l_tax REAL,
                    l_returnflag TEXT,
                    l_linestatus TEXT,
                    l_shipdate TEXT,
                    l_commitdate TEXT,
                    l_receiptdate TEXT,
                    PRIMARY KEY (l_orderkey, l_linenumber),
                    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey)
                )"""
            ]
        else:  # PostgreSQL
            schema_commands = [
                """CREATE TABLE customers (
                    c_custkey SERIAL PRIMARY KEY,
                    c_name VARCHAR(255) NOT NULL,
                    c_address VARCHAR(255),
                    c_nationkey INTEGER,
                    c_phone VARCHAR(20),
                    c_acctbal DECIMAL(10,2),
                    c_mktsegment VARCHAR(50)
                )""",
                
                """CREATE TABLE orders (
                    o_orderkey SERIAL PRIMARY KEY,
                    o_custkey INTEGER REFERENCES customers(c_custkey),
                    o_orderstatus VARCHAR(1),
                    o_totalprice DECIMAL(10,2),
                    o_orderdate DATE,
                    o_orderpriority VARCHAR(20),
                    o_clerk VARCHAR(20),
                    o_shippriority INTEGER
                )""",
                
                """CREATE TABLE lineitem (
                    l_orderkey INTEGER REFERENCES orders(o_orderkey),
                    l_partkey INTEGER,
                    l_suppkey INTEGER,
                    l_linenumber INTEGER,
                    l_quantity DECIMAL(10,2),
                    l_extendedprice DECIMAL(10,2),
                    l_discount DECIMAL(10,2),
                    l_tax DECIMAL(10,2),
                    l_returnflag VARCHAR(1),
                    l_linestatus VARCHAR(1),
                    l_shipdate DATE,
                    l_commitdate DATE,
                    l_receiptdate DATE,
                    PRIMARY KEY (l_orderkey, l_linenumber)
                )"""
            ]
        
        try:
            # Drop existing tables
            for command in drop_commands:
                try:
                    cursor.execute(command)
                except:
                    pass  # Ignore if table doesn't exist
            
            # Create new tables
            for command in schema_commands:
                cursor.execute(command)
            
            if self.db_type == "sqlite":
                self.connection.commit()
            
            logger.info("Sample tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create sample tables: {e}")
            raise
        finally:
            cursor.close()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from query history."""
        if not self.query_history:
            return {"message": "No queries executed yet"}
        
        successful_queries = [q for q in self.query_history if not q.error]
        failed_queries = [q for q in self.query_history if q.error]
        
        if successful_queries:
            execution_times = [q.execution_time for q in successful_queries]
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
        else:
            avg_time = min_time = max_time = 0
        
        return {
            "total_queries": len(self.query_history),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "avg_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time
        }