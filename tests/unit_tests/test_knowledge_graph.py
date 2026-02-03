"""
Test the knowledge graph system.
"""
import sys
import os
sys.path.append('src')

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.utils.logging import logger

def test_knowledge_graph():
    """Test knowledge graph functionality."""
    
    logger.info("=== Testing Knowledge Graph System ===")
    
    # Create database with sample data
    db = DatabaseSimulator(db_path="test_kg.db")
    
    try:
        # Setup database
        db.connect()
        db.create_sample_tables()
        
        # Insert test data
        test_data = [
            "INSERT INTO customers VALUES (1, 'Alice', '123 Main', 1, '555-0001', 1000, 'BUILDING')",
            "INSERT INTO customers VALUES (2, 'Bob', '456 Oak', 2, '555-0002', 2000, 'AUTOMOBILE')",
            "INSERT INTO customers VALUES (3, 'Carol', '789 Pine', 1, '555-0003', 1500, 'BUILDING')",
            
            "INSERT INTO orders VALUES (1, 1, 'O', 150, '2023-01-15', '1-URGENT', 'Clerk001', 0)",
            "INSERT INTO orders VALUES (2, 2, 'O', 300, '2023-01-16', '2-HIGH', 'Clerk002', 0)",
            "INSERT INTO orders VALUES (3, 1, 'O', 225, '2023-01-17', '1-URGENT', 'Clerk001', 0)",
            "INSERT INTO orders VALUES (4, 3, 'O', 400, '2023-01-18', '3-MEDIUM', 'Clerk003', 0)",
            "INSERT INTO orders VALUES (5, 2, 'O', 175, '2023-01-19', '2-HIGH', 'Clerk002', 0)",
            
            "INSERT INTO lineitem VALUES (1, 101, 201, 1, 5, 50, 0.05, 0.08, 'N', 'O', '2023-01-20', '2023-01-18', '2023-01-22')",
            "INSERT INTO lineitem VALUES (2, 102, 202, 1, 3, 75, 0.03, 0.06, 'N', 'O', '2023-01-21', '2023-01-19', '2023-01-23')",
            "INSERT INTO lineitem VALUES (3, 103, 203, 1, 7, 140, 0.07, 0.09, 'N', 'O', '2023-01-22', '2023-01-20', '2023-01-24')",
            "INSERT INTO lineitem VALUES (4, 104, 204, 1, 2, 40, 0.02, 0.05, 'N', 'O', '2023-01-23', '2023-01-21', '2023-01-25')",
            "INSERT INTO lineitem VALUES (5, 105, 205, 1, 6, 120, 0.06, 0.07, 'N', 'O', '2023-01-24', '2023-01-22', '2023-01-26')",
            "INSERT INTO lineitem VALUES (1, 106, 206, 2, 4, 80, 0.04, 0.06, 'N', 'O', '2023-01-25', '2023-01-23', '2023-01-27')",
            "INSERT INTO lineitem VALUES (2, 107, 207, 2, 8, 160, 0.08, 0.10, 'N', 'O', '2023-01-26', '2023-01-24', '2023-01-28')",
            "INSERT INTO lineitem VALUES (3, 108, 208, 2, 1, 20, 0.01, 0.04, 'N', 'O', '2023-01-27', '2023-01-25', '2023-01-29')"
        ]
        
        for query in test_data:
            result = db.execute_query(query)
            if result.error:
                logger.error(f"Failed to insert data: {result.error}")
                return False
        
        logger.info("Test data inserted successfully")
        
        # Create and build knowledge graph
        kg = DatabaseSchemaKG(db_type="sqlite")
        kg.build_from_database(db.connection)
        logger.info("Knowledge graph built from database")
        
        # Test 1: Check table information
        customers_info = kg.get_table_info("customers")
        if customers_info:
            logger.info(f"Customers table: {len(customers_info.columns)} columns, {customers_info.row_count} rows")
            assert customers_info.row_count == 3, f"Expected 3 customers, got {customers_info.row_count}"
            assert len(customers_info.columns) == 7, f"Expected 7 columns, got {len(customers_info.columns)}"
            logger.info("Test 1 passed: Table information retrieval")
        else:
            logger.error("Could not retrieve customers table info")
            return False
        
        # Test 2: Check orders table
        orders_info = kg.get_table_info("orders")
        if orders_info:
            logger.info(f"Orders table: {len(orders_info.columns)} columns, {orders_info.row_count} rows")
            assert orders_info.row_count == 5, f"Expected 5 orders, got {orders_info.row_count}"
            assert orders_info.foreign_keys is not None, "Orders should have foreign keys"
            logger.info("Test 2 passed: Orders table with foreign keys")
        else:
            logger.error("Could not retrieve orders table info")
            return False
        
        # Test 3: Check lineitem table
        lineitem_info = kg.get_table_info("lineitem")
        if lineitem_info:
            logger.info(f"Lineitem table: {len(lineitem_info.columns)} columns, {lineitem_info.row_count} rows")
            assert lineitem_info.row_count == 8, f"Expected 8 lineitems, got {lineitem_info.row_count}"
            logger.info("Test 3 passed: Lineitem table information")
        else:
            logger.error("Could not retrieve lineitem table info")
            return False
        
        # Test 4: Check relationships
        related_to_customers = kg.get_related_tables("customers")
        logger.info(f"Tables related to customers: {related_to_customers}")
        assert "orders" in related_to_customers, "Orders should be related to customers"
        logger.info("Test 4 passed: Table relationships")
        
        # Test 5: Check join information
        join_info = kg.get_join_info("orders", "customers")
        if join_info:
            logger.info(f"Join info orders->customers: {join_info.left_column} -> {join_info.right_column}")
            assert join_info.left_column == "o_custkey" or join_info.right_column == "o_custkey", "Should involve o_custkey"
            logger.info("Test 5 passed: Join information retrieval")
        else:
            logger.error("No join found between orders and customers")
            return False
        
        # Test 6: Test join order suggestion
        test_tables = ["customers", "orders", "lineitem"]
        suggested_order = kg.suggest_join_order(test_tables)
        logger.info(f"Suggested join order for {test_tables}: {suggested_order}")
        
        # Should start with smallest table (customers: 3 rows)
        assert suggested_order[0] == "customers", f"Should start with customers (smallest), got {suggested_order[0]}"
        assert len(suggested_order) == 3, f"Should return 3 tables, got {len(suggested_order)}"
        logger.info("Test 6 passed: Join order suggestion")
        
        # Test 7: Check all tables are present
        all_tables = list(kg.tables.keys())
        expected_tables = {"customers", "orders", "lineitem"}
        actual_tables = set(all_tables)
        assert expected_tables.issubset(actual_tables), f"Missing tables. Expected {expected_tables}, got {actual_tables}"
        logger.info("Test 7 passed: All expected tables present")
        
        # Test 8: Check relationships count
        assert len(kg.relationships) >= 2, f"Should have at least 2 relationships, got {len(kg.relationships)}"
        logger.info("Test 8 passed: Sufficient relationships found")
        
        # Test 9: Test edge cases
        non_existent_table = kg.get_table_info("non_existent")
        assert non_existent_table is None, "Should return None for non-existent table"
        
        empty_join_order = kg.suggest_join_order([])
        assert empty_join_order == [], "Should return empty list for empty input"
        
        single_table_order = kg.suggest_join_order(["customers"])
        assert single_table_order == ["customers"], "Should return same table for single table input"
        logger.info("Test 9 passed: Edge cases handled correctly")
        
        # Test 10: Print summary (visual verification)
        kg.print_summary()
        logger.info("Test 10 passed: Summary printed successfully")
        
        logger.info("All knowledge graph tests completed successfully")
        
    except AssertionError as e:
        logger.error(f"Test assertion failed: {e}")
        pytest.fail(f"Test assertion failed: {e}")
    except Exception as e:
        logger.error(f"Knowledge graph test failed: {e}")
        pytest.fail(f"Knowledge graph test failed: {e}")
        
    finally:
        # Clean up
        db.disconnect()
        if os.path.exists("test_kg.db"):
            os.remove("test_kg.db")

def test_database_type_validation():
    """Test database type validation."""
    logger.info("=== Testing Database Type Validation ===")
    
    try:
        # Test valid database types
        sqlite_kg = DatabaseSchemaKG(db_type="sqlite")
        assert sqlite_kg.db_type == "sqlite"
        logger.info("SQLite database type validation passed")
        
        postgresql_kg = DatabaseSchemaKG(db_type="postgresql")
        assert postgresql_kg.db_type == "postgresql"
        logger.info("PostgreSQL database type validation passed")
        
        # Test invalid database type
        try:
            invalid_kg = DatabaseSchemaKG(db_type="invalid_db")
            invalid_kg._get_metadata_extractor()  # This should raise an error
            logger.error("Should have raised error for invalid database type")
            pytest.fail("Should have raised error for invalid database type")
        except ValueError as e:
            logger.info(f"Correctly caught invalid database type error: {e}")
        
        logger.info("Database type validation tests passed")
        
    except Exception as e:
        logger.error(f"Database type validation test failed: {e}")
        pytest.fail(f"Database type validation test failed: {e}")

if __name__ == "__main__":
    # Run main functionality test
    success1 = test_knowledge_graph()
    
    # Run validation test
    success2 = test_database_type_validation()
    
    if success1 and success2:
        print("\nAll Knowledge Graph tests passed!")
        print("Knowledge Graph system is working correctly.")
        print("Next: We'll create the baseline optimizer.")
    else:
        print("\nSome Knowledge Graph tests failed.")
        print("Please check the errors above.")