"""
Pytest configuration and fixtures for intelligent database optimizer tests.
"""
import pytest
import sys
import os
import tempfile
import shutil
from typing import Generator

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.agents.rl_environment import QueryOptimizationEnv
from src.agents.dqn_agent import MultiAgentDQN


@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_database(temp_dir: str) -> Generator[DatabaseSimulator, None, None]:
    """Create test database with sample data."""
    db_path = os.path.join(temp_dir, "test.db")
    db = DatabaseSimulator(db_type="sqlite", db_path=db_path)
    
    # Initialize database
    db.connect()
    db.create_sample_tables()
    
    # Insert test data
    _populate_test_data(db)
    
    yield db
    
    # Cleanup
    if hasattr(db, 'connection') and db.connection:
        db.connection.close()


@pytest.fixture(scope="function")
def db_simulator(test_database: DatabaseSimulator) -> DatabaseSimulator:
    """Alias fixture for backward compatibility."""
    return test_database


@pytest.fixture(scope="function")
def knowledge_graph(test_database: DatabaseSimulator) -> DatabaseSchemaKG:
    """Create knowledge graph from test database."""
    kg = DatabaseSchemaKG(db_type="sqlite")
    kg.build_from_database(test_database.connection)
    return kg


@pytest.fixture(scope="function")
def rl_environment(test_database: DatabaseSimulator, knowledge_graph: DatabaseSchemaKG) -> QueryOptimizationEnv:
    """Create RL environment for testing."""
    return QueryOptimizationEnv(test_database, knowledge_graph)


@pytest.fixture(scope="function")
def dqn_system() -> MultiAgentDQN:
    """Create multi-agent DQN system."""
    return MultiAgentDQN()


def _populate_test_data(db: DatabaseSimulator) -> None:
    """Populate database with test data."""
    # Insert customers
    customers_data = [
        ("Customer_1", "Address_1", 1, "555-0001", 1500.00, "BUILDING"),
        ("Customer_2", "Address_2", 2, "555-0002", 2500.00, "AUTOMOBILE"),
        ("Customer_3", "Address_3", 3, "555-0003", 3500.00, "MACHINERY"),
    ]
    
    for customer in customers_data:
            query = "INSERT INTO customers (c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment) VALUES (?, ?, ?, ?, ?, ?)"
            cursor = db.connection.cursor()
            cursor.execute(query, customer)
            db.connection.commit()
    # Insert orders
    orders_data = [
        (1, "O", 1000.00, "2023-01-01", "1-URGENT", "Clerk_1", 0),
        (2, "O", 2000.00, "2023-01-02", "2-HIGH", "Clerk_2", 0),
        (3, "O", 1500.00, "2023-01-03", "3-MEDIUM", "Clerk_3", 0),
    ]
    
    for order in orders_data:
        query = "INSERT INTO orders (o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority) VALUES (?, ?, ?, ?, ?, ?, ?)"
        cursor = db.connection.cursor()
        cursor.execute(query, order)
        db.connection.commit()
    
    # Insert lineitems
    lineitems_data = [
        (1, 1, 1, 1, 10, 100.00, 0.05, 0.08, "N", "O", "2023-01-10", "2023-01-05", "2023-01-12"),
        (2, 2, 2, 1, 20, 200.00, 0.10, 0.08, "N", "O", "2023-01-15", "2023-01-10", "2023-01-18"),
        (3, 3, 3, 1, 15, 150.00, 0.07, 0.08, "N", "O", "2023-01-20", "2023-01-15", "2023-01-22"),
    ]
    
    for lineitem in lineitems_data:
        query = "INSERT INTO lineitem (l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        cursor = db.connection.cursor()
        cursor.execute(query, lineitem)
        db.connection.commit()