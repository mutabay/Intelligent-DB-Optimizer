"""
Test PostgreSQL connection with your credentials.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.database_environment.db_simulator import DatabaseSimulator
from src.utils.logging import logger

def test_postgresql_connection():
    """Test PostgreSQL connection."""
    
    logger.info("=== Testing PostgreSQL Connection ===")
    
    # Test PostgreSQL connection
    db = DatabaseSimulator(db_type="postgresql")
    
    try:
        # Test connection
        db.connect()
        logger.info("PostgreSQL connection successful")
        
        # Test simple query
        result = db.execute_query("SELECT version();")
        if result.error:
            logger.error(f"Version query failed: {result.error}")
            return False
        else:
            logger.info("PostgreSQL version query successful")
        
        # Test table creation
        db.create_sample_tables()
        logger.info("Sample tables created successfully")
        
        # Test a simple insert/select
        test_queries = [
            "INSERT INTO customers (c_name, c_mktsegment) VALUES ('Test Customer', 'BUILDING')",
            "SELECT COUNT(*) FROM customers"
        ]
        
        for query in test_queries:
            result = db.execute_query(query)
            if result.error:
                logger.error(f"Test query failed: {result.error}")
                pytest.fail(f"Test query failed: {result.error}")
        
        logger.info("All PostgreSQL tests passed")
        
    except Exception as e:
        logger.error(f"PostgreSQL test failed: {e}")
        pytest.fail(f"PostgreSQL test failed: {e}")
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = test_postgresql_connection()
    if success:
        print("PostgreSQL connection is working!")
    else:
        print("PostgreSQL connection failed. Check your credentials in .env file.")