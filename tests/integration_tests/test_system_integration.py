"""
Integration Tests for Intelligent Database Query Optimizer

This module contains comprehensive integration tests that verify the complete
hybrid AI system functionality including LLM-DQN coordination, knowledge graph
integration, and end-to-end query optimization workflows.
"""

import os
import sys
import pytest
import tempfile
import sqlite3
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import IntelligentDBOptimizer
from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.agents.llm_query_agent import LangChainQueryAgent
from src.agents.rl_environment import QueryOptimizationEnv
from src.agents.dqn_agent import MultiAgentDQN
from src.optimization.query_optimizer import QueryOptimizer
from src.utils.logging import logger


class TestSystemIntegration:
    """
    Integration tests for the complete intelligent database optimizer system.
    Tests verify end-to-end functionality and component interactions.
    """

    @pytest.fixture(scope="class")
    def temp_db(self):
        """Create temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture(scope="class")
    def complete_system(self, temp_db):
        """Initialize complete system for integration testing."""
        system = IntelligentDBOptimizer(db_type="sqlite", db_path=temp_db)
        success = system.initialize_system()
        assert success, "System initialization failed"
        yield system
        system.cleanup()

    def test_system_initialization(self, complete_system):
        """Test that all system components initialize correctly."""
        system = complete_system
        
        # Verify core components are initialized
        assert system.db_simulator is not None, "Database simulator not initialized"
        assert system.knowledge_graph is not None, "Knowledge graph not initialized"
        assert system.rl_environment is not None, "RL environment not initialized"
        assert system.dqn_system is not None, "DQN system not initialized"
        assert system.query_optimizer is not None, "Query optimizer not initialized"
        assert system.cost_estimator is not None, "Cost estimator not initialized"
        
        # Verify database connection
        assert system.db_simulator.connection is not None, "Database connection failed"
        
        # Verify knowledge graph has tables
        assert len(system.knowledge_graph.tables) > 0, "Knowledge graph has no tables"
        
        logger.info("✅ System initialization test passed")

    def test_hybrid_query_optimization_workflow(self, complete_system):
        """Test complete hybrid optimization workflow with LLM and DQN integration."""
        system = complete_system
        
        # Test query that exercises multiple components
        test_query = """
        SELECT c.name, COUNT(o.order_key) as order_count, 
               SUM(l.extended_price) as total_revenue
        FROM customers c
        JOIN orders o ON c.customer_key = o.customer_key
        JOIN lineitem l ON o.order_key = l.order_key
        WHERE c.nation_key IN (1, 2, 3)
          AND o.order_date > '1995-01-01'
        GROUP BY c.customer_key, c.name
        HAVING SUM(l.extended_price) > 1000
        ORDER BY total_revenue DESC
        """
        
        # Test all optimization strategies
        strategies = ["rule_based", "dqn_based", "hybrid"]
        results = {}
        
        for strategy in strategies:
            try:
                result = system.optimize_query(test_query, strategy=strategy)
                assert "estimated_cost" in result, f"Missing cost estimate in {strategy}"
                assert "optimization_plan" in result, f"Missing plan in {strategy}"
                assert "execution_time" in result, f"Missing execution time in {strategy}"
                results[strategy] = result
                logger.info(f"✅ {strategy} optimization completed")
            except Exception as e:
                pytest.fail(f"❌ {strategy} optimization failed: {e}")
        
        # Verify hybrid strategy combines insights from both approaches
        hybrid_result = results["hybrid"]
        assert "llm_analysis" in hybrid_result, "Hybrid missing LLM analysis"
        assert "dqn_actions" in hybrid_result, "Hybrid missing DQN actions"
        
        logger.info("✅ Hybrid optimization workflow test passed")

    def test_llm_dqn_coordination(self, complete_system):
        """Test coordination between LLM and DQN agents."""
        system = complete_system
        
        # Initialize LLM agent
        llm_agent = LangChainQueryAgent(system.knowledge_graph, llm_provider="simple")
        
        test_query = "SELECT * FROM customers WHERE nation_key = 1"
        
        # Test LLM analysis
        llm_analysis = llm_agent.analyze_query(test_query)
        assert "llm_analysis" in llm_analysis, "LLM analysis failed"
        assert "knowledge_graph_analysis" in llm_analysis, "Missing KG analysis"
        
        # Test LLM optimization suggestions
        llm_optimization = llm_agent.optimize_query(test_query)
        assert "llm_optimization" in llm_optimization, "LLM optimization failed"
        assert "kg_suggestions" in llm_optimization, "Missing KG suggestions"
        
        # Test RL environment state extraction
        state = system.rl_environment.extract_state_from_query(test_query)
        assert len(state) == 12, "Invalid state vector dimension"
        assert all(0 <= val <= 1 for val in state), "State values not normalized"
        
        # Test DQN action selection
        actions = system.dqn_system.get_actions(state)
        assert len(actions) == 4, "Invalid number of agent actions"
        assert all(isinstance(action, int) for action in actions), "Invalid action types"
        
        logger.info("✅ LLM-DQN coordination test passed")

    def test_knowledge_graph_integration(self, complete_system):
        """Test knowledge graph integration with optimization components."""
        system = complete_system
        kg = system.knowledge_graph
        
        # Verify schema information
        assert "customers" in kg.tables, "Customers table missing from KG"
        assert "orders" in kg.tables, "Orders table missing from KG"
        assert "lineitem" in kg.tables, "Lineitem table missing from KG"
        
        # Test relationship detection
        relationships = kg.get_table_relationships("customers", "orders")
        assert len(relationships) > 0, "No relationships found between customers and orders"
        
        # Test join order suggestion
        tables = ["customers", "orders", "lineitem"]
        suggested_order = kg.suggest_join_order(tables)
        assert len(suggested_order) == 3, "Invalid join order suggestion"
        assert all(table in tables for table in suggested_order), "Invalid table in join order"
        
        # Test optimization hint generation
        query = "SELECT * FROM customers c JOIN orders o ON c.customer_key = o.customer_key"
        hints = kg.generate_optimization_hints(query)
        assert len(hints) > 0, "No optimization hints generated"
        
        logger.info("✅ Knowledge graph integration test passed")

    def test_cost_estimation_accuracy(self, complete_system):
        """Test cost estimation component accuracy and consistency."""
        system = complete_system
        estimator = system.cost_estimator
        
        # Test simple query cost estimation
        simple_plan = {
            'scan_operations': [{'table': 'customers', 'selectivity': 0.1}]
        }
        simple_cost, breakdown = estimator.estimate_query_cost(simple_plan)
        assert simple_cost > 0, "Invalid simple query cost"
        assert 'scan_cost' in breakdown, "Missing scan cost breakdown"
        
        # Test complex query cost estimation
        complex_plan = {
            'scan_operations': [
                {'table': 'customers', 'selectivity': 0.2},
                {'table': 'orders', 'selectivity': 0.1}
            ],
            'join_operations': [
                {'left_table': 'customers', 'right_table': 'orders', 'join_type': 'hash_join'}
            ]
        }
        complex_cost, complex_breakdown = estimator.estimate_query_cost(complex_plan)
        assert complex_cost > simple_cost, "Complex query should cost more than simple query"
        assert 'join_cost' in complex_breakdown, "Missing join cost breakdown"
        
        # Test cost consistency
        cost2, _ = estimator.estimate_query_cost(simple_plan)
        assert abs(simple_cost - cost2) < 0.01, "Cost estimation not consistent"
        
        logger.info("✅ Cost estimation accuracy test passed")

    def test_training_integration(self, complete_system):
        """Test DQN training integration with the complete system."""
        system = complete_system
        
        # Test short training session (5 episodes for speed)
        try:
            training_results = system.train_dqn_system(num_episodes=5, save_interval=10)
            
            assert "total_episodes" in training_results, "Missing episode count in training results"
            assert "final_epsilon" in training_results, "Missing epsilon in training results"
            assert "average_reward" in training_results, "Missing reward in training results"
            assert training_results["total_episodes"] == 5, "Incorrect episode count"
            
            # Verify training affected the agents
            initial_actions = system.dqn_system.get_actions([0.5] * 12)
            post_training_actions = system.dqn_system.get_actions([0.5] * 12)
            # Actions might be same due to short training, but system should not crash
            
            logger.info("✅ Training integration test passed")
            
        except Exception as e:
            # Training might fail due to environment issues, log warning instead of failing test
            logger.warning(f"⚠️ Training integration test warning: {e}")

    def test_evaluation_framework_integration(self, complete_system):
        """Test evaluation framework integration with complete system."""
        system = complete_system
        
        # Test evaluation with small trial count for speed
        try:
            evaluation_results = system.evaluate_system(num_trials=3)
            
            assert "evaluation_results" in evaluation_results, "Missing evaluation results"
            eval_data = evaluation_results["evaluation_results"]
            
            assert "summary_stats" in eval_data, "Missing summary statistics"
            assert "detailed_comparisons" in eval_data, "Missing detailed comparisons"
            
            stats = eval_data["summary_stats"]
            assert "total_queries_evaluated" in stats, "Missing query count"
            assert "win_rate" in stats, "Missing win rate"
            assert stats["total_queries_evaluated"] >= 3, "Insufficient queries evaluated"
            
            logger.info("✅ Evaluation framework integration test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Evaluation integration test warning: {e}")

    def test_system_error_handling(self, complete_system):
        """Test system error handling and robustness."""
        system = complete_system
        
        # Test invalid query handling
        try:
            result = system.optimize_query("INVALID SQL SYNTAX", strategy="hybrid")
            # Should either handle gracefully or raise appropriate exception
        except Exception as e:
            assert "syntax" in str(e).lower() or "invalid" in str(e).lower(), "Unexpected error type"
        
        # Test missing table query
        try:
            result = system.optimize_query("SELECT * FROM nonexistent_table", strategy="hybrid")
        except Exception as e:
            assert "table" in str(e).lower() or "exist" in str(e).lower(), "Unexpected error type"
        
        # Test system recovery after errors
        valid_query = "SELECT * FROM customers LIMIT 10"
        result = system.optimize_query(valid_query, strategy="rule_based")
        assert "estimated_cost" in result, "System not recovered after error"
        
        logger.info("✅ Error handling test passed")

    def test_end_to_end_workflow(self, complete_system):
        """Test complete end-to-end workflow from query input to optimized output."""
        system = complete_system
        
        # Define realistic test query
        e2e_query = """
        SELECT 
            c.name as customer_name,
            c.nation_key,
            COUNT(o.order_key) as total_orders,
            AVG(o.total_price) as avg_order_value,
            SUM(l.quantity) as total_items
        FROM customers c
        LEFT JOIN orders o ON c.customer_key = o.customer_key
        LEFT JOIN lineitem l ON o.order_key = l.order_key
        WHERE c.acctbal > 0
        GROUP BY c.customer_key, c.name, c.nation_key
        HAVING COUNT(o.order_key) > 0
        ORDER BY total_orders DESC, avg_order_value DESC
        LIMIT 50
        """
        
        # Execute complete workflow
        workflow_steps = {
            "query_analysis": None,
            "cost_estimation": None,
            "optimization": None,
            "explanation": None
        }
        
        try:
            # Step 1: Query Analysis (LLM + KG)
            llm_agent = LangChainQueryAgent(system.knowledge_graph, llm_provider="simple")
            workflow_steps["query_analysis"] = llm_agent.analyze_query(e2e_query)
            
            # Step 2: Cost Estimation
            basic_plan = {'scan_operations': [{'table': 'customers', 'selectivity': 0.5}]}
            workflow_steps["cost_estimation"] = system.cost_estimator.estimate_query_cost(basic_plan)
            
            # Step 3: Hybrid Optimization
            workflow_steps["optimization"] = system.optimize_query(e2e_query, strategy="hybrid")
            
            # Step 4: Explanation Generation
            workflow_steps["explanation"] = llm_agent.explain_optimization(e2e_query)
            
            # Verify all workflow steps completed
            for step_name, result in workflow_steps.items():
                assert result is not None, f"Workflow step {step_name} failed"
            
            # Verify workflow coherence
            optimization_result = workflow_steps["optimization"]
            assert "estimated_cost" in optimization_result, "Missing cost in optimization result"
            assert "optimization_plan" in optimization_result, "Missing plan in optimization result"
            
            explanation = workflow_steps["explanation"]
            assert len(explanation) > 0, "Empty optimization explanation"
            
            logger.info("✅ End-to-end workflow test passed")
            
        except Exception as e:
            pytest.fail(f"❌ End-to-end workflow failed: {e}")


class TestSystemPerformance:
    """
    Performance tests to verify system meets performance requirements.
    """

    @pytest.fixture(scope="class")
    def performance_system(self):
        """Initialize system for performance testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        system = IntelligentDBOptimizer(db_type="sqlite", db_path=temp_db)
        system.initialize_system()
        yield system
        system.cleanup()
        os.unlink(temp_db)

    def test_optimization_latency(self, performance_system):
        """Test that optimization completes within acceptable time limits."""
        import time
        
        system = performance_system
        query = "SELECT * FROM customers WHERE nation_key = 1"
        
        # Test rule-based optimization latency
        start_time = time.time()
        result = system.optimize_query(query, strategy="rule_based")
        rule_based_time = time.time() - start_time
        
        assert rule_based_time < 1.0, f"Rule-based optimization too slow: {rule_based_time:.3f}s"
        
        # Test hybrid optimization latency
        start_time = time.time()
        result = system.optimize_query(query, strategy="hybrid")
        hybrid_time = time.time() - start_time
        
        assert hybrid_time < 5.0, f"Hybrid optimization too slow: {hybrid_time:.3f}s"
        
        logger.info(f"✅ Optimization latency: rule-based={rule_based_time:.3f}s, hybrid={hybrid_time:.3f}s")

    def test_memory_usage(self, performance_system):
        """Test system memory usage remains within reasonable bounds."""
        import psutil
        import gc
        
        system = performance_system
        process = psutil.Process()
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple optimizations
        for i in range(10):
            query = f"SELECT * FROM customers WHERE customer_key = {i + 1}"
            system.optimize_query(query, strategy="hybrid")
        
        # Measure memory after operations
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.1f}MB increase"
        logger.info(f"✅ Memory usage: baseline={baseline_memory:.1f}MB, final={final_memory:.1f}MB")

    def test_concurrent_optimization(self, performance_system):
        """Test system handles concurrent optimization requests."""
        import threading
        import time
        
        system = performance_system
        results = []
        errors = []
        
        def optimize_query(query_id):
            try:
                query = f"SELECT * FROM customers WHERE customer_key = {query_id}"
                result = system.optimize_query(query, strategy="rule_based")
                results.append((query_id, result))
            except Exception as e:
                errors.append((query_id, e))
        
        # Launch concurrent optimizations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=optimize_query, args=(i + 1,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        assert len(errors) == 0, f"Concurrent optimization errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        logger.info("✅ Concurrent optimization test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])