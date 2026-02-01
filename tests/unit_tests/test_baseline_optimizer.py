"""
Test the baseline rule-based optimizer.
"""
import sys
import os
sys.path.append('src')
sys.path.append('evaluation')

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from evaluation.baselines.rule_based_optimizer import RuleBasedOptimizer, BaselineEvaluator
from src.utils.logging import logger

def test_baseline_optimizer():
    """Test the baseline rule-based optimizer."""
    
    logger.info("=== Testing Baseline Rule-Based Optimizer ===")
    
    # Setup database and knowledge graph
    db = DatabaseSimulator(db_path="test_baseline.db")
    
    try:
        # Create database with test data
        db.connect()
        db.create_sample_tables()
        
        # Insert substantial test data
        test_data = []
        
        # Insert customers (50 customers)
        for i in range(1, 51):
            segment = "BUILDING" if i % 3 == 0 else "AUTOMOBILE" if i % 3 == 1 else "MACHINERY"
            test_data.append(f"INSERT INTO customers VALUES ({i}, 'Customer_{i}', 'Address_{i}', {i % 25}, '555-{i:04d}', {1000 + i * 50}, '{segment}')")
        
        # Insert orders (100 orders)
        for i in range(1, 101):
            cust_id = (i % 50) + 1
            status = "O" if i % 2 == 0 else "F"
            priority = f"{(i % 5) + 1}-URGENT" if i % 5 == 0 else f"{(i % 5) + 1}-HIGH"
            test_data.append(f"INSERT INTO orders VALUES ({i}, {cust_id}, '{status}', {100 + i * 25}, '2023-01-{(i % 28) + 1:02d}', '{priority}', 'Clerk_{i % 10}', 0)")
        
        # Insert lineitems (200 lineitems)
        for i in range(1, 201):
            order_id = (i % 100) + 1
            line_num = (i % 7) + 1
            test_data.append(f"INSERT INTO lineitem VALUES ({order_id}, {100 + i}, {200 + i}, {line_num}, {5 + (i % 10)}, {50 + i * 2}, 0.{i % 10:02d}, 0.{(i % 8) + 1:02d}, 'N', 'O', '2023-01-{(i % 28) + 1:02d}', '2023-01-{(i % 28) + 1:02d}', '2023-01-{(i % 28) + 1:02d}')")
        
        # Execute all insert statements
        for query in test_data:
            result = db.execute_query(query)
            if result.error:
                logger.error(f"Failed to insert data: {result.error}")
                return False
        
        logger.info(f"Inserted test data: 50 customers, 100 orders, 200 lineitems")
        
        # Build knowledge graph
        kg = DatabaseSchemaKG(db_type="sqlite")
        kg.build_from_database(db.connection)
        
        # Create optimizer
        optimizer = RuleBasedOptimizer(kg)
        
        # Test individual query optimization
        test_query = """
        SELECT c.c_name, o.o_totalprice, l.l_quantity
        FROM customers c
        JOIN orders o ON c.c_custkey = o.o_custkey
        JOIN lineitem l ON o.o_orderkey = l.l_orderkey
        WHERE c.c_mktsegment = 'BUILDING'
        AND o.o_totalprice > 1000
        """
        
        optimization_result = optimizer.optimize_query(test_query)
        
        logger.info(f"Original query length: {len(optimization_result.original_query)} chars")
        logger.info(f"Optimized query length: {len(optimization_result.optimized_query)} chars")
        logger.info(f"Optimization strategy: {optimization_result.optimization_strategy}")
        logger.info(f"Estimated improvement: {optimization_result.estimated_improvement:.1%}")
        
        print("\n=== Query Optimization Example ===")
        print("Original Query:")
        print(optimization_result.original_query.strip())
        print("\nOptimized Query:")
        print(optimization_result.optimized_query.strip())
        print(f"\nStrategy: {optimization_result.optimization_strategy}")
        print(f"Estimated Improvement: {optimization_result.estimated_improvement:.1%}")
        
        # Test baseline evaluator
        evaluator = BaselineEvaluator(db, kg)
        
        # Define test queries for evaluation
        evaluation_queries = [
            "SELECT COUNT(*) FROM customers WHERE c_mktsegment = 'BUILDING'",
            "SELECT c.c_name, COUNT(*) FROM customers c JOIN orders o ON c.c_custkey = o.o_custkey GROUP BY c.c_name",
            "SELECT o.o_orderkey, o.o_totalprice FROM orders o WHERE o.o_totalprice > 2000",
            """SELECT c.c_name, o.o_totalprice 
               FROM customers c 
               JOIN orders o ON c.c_custkey = o.o_custkey 
               WHERE c.c_mktsegment = 'AUTOMOBILE'""",
            """SELECT c.c_name, o.o_totalprice, l.l_quantity
               FROM customers c
               JOIN orders o ON c.c_custkey = o.o_custkey
               JOIN lineitem l ON o.o_orderkey = l.l_orderkey
               WHERE c.c_mktsegment = 'BUILDING'"""
        ]
        
        # Run evaluation
        metrics = evaluator.evaluate_baseline_performance(evaluation_queries)
        evaluator.print_evaluation_summary(metrics)
        
        # Verify results
        assert metrics["total_queries"] == 5, f"Expected 5 queries, got {metrics['total_queries']}"
        assert metrics["successful_executions"] > 0, "Should have successful executions"
        assert "avg_original_time" in metrics, "Should have average original time"
        
        logger.info("Baseline optimizer test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Baseline optimizer test failed: {e}")
        return False
        
    finally:
        # Clean up
        db.disconnect()
        if os.path.exists("test_baseline.db"):
            os.remove("test_baseline.db")

if __name__ == "__main__":
    success = test_baseline_optimizer()
    if success:
        print("\nBaseline Optimizer is working correctly!")
        print("This will serve as our comparison benchmark.")
        print("Next: We'll start building the LLM-based query understanding agent.")
    else:
        print("\nBaseline optimizer test failed.")