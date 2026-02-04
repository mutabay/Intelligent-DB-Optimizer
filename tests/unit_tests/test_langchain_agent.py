"""
Test LangChain-based query agent.
"""

import sys
import os
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.agents.llm_query_agent import LangChainQueryAgent
from src.utils.logging import logger

def test_enhanced_llm_agent():
    """Test LLM agent with different providers."""
    
    logger.info("=== Testing LLM Query Agent ===")
    
    # Setup
    db = DatabaseSimulator(db_type="sqlite", db_path="test_enhanced_llm.db")
    
    try:
        db.connect()
        db.create_sample_tables()
        
        # Add sample data
        sample_data = [
            "INSERT INTO customers (c_name, c_mktsegment) VALUES ('Customer 1', 'BUILDING')",
            "INSERT INTO customers (c_name, c_mktsegment) VALUES ('Customer 2', 'AUTOMOBILE')",
            "INSERT INTO orders (o_custkey, o_totalprice) VALUES (1, 1500)",
            "INSERT INTO orders (o_custkey, o_totalprice) VALUES (2, 2500)",
            "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice) VALUES (1, 5, 100)",
            "INSERT INTO lineitem (l_orderkey, l_quantity, l_extendedprice) VALUES (2, 3, 150)"
        ]
        
        for query in sample_data:
            db.execute_query(query)
        
        # Build knowledge graph
        kg = DatabaseSchemaKG(db_type="sqlite")
        kg.build_from_database(db.connection)
        
        # Test different LLM providers
        providers_to_test = ["simple", "ollama"]  # Add "openai" if you have API key
        
        for provider in providers_to_test:
            print(f"\n{'='*60}")
            print(f"Testing with LLM Provider: {provider.upper()}")
            print(f"{'='*60}")
            
            # Create agent with specific provider
            agent = LangChainQueryAgent(kg, llm_provider=provider)
            
            # Test complex query
            test_query = """
            SELECT c.c_name, c.c_mktsegment, o.o_totalprice, l.l_quantity
            FROM customers c
            JOIN orders o ON c.c_custkey = o.o_custkey
            JOIN lineitem l ON o.o_orderkey = l.l_orderkey
            WHERE c.c_mktsegment = 'BUILDING'
            AND o.o_totalprice > 1000
            ORDER BY o.o_totalprice DESC
            """
            
            print(f"Testing query: {test_query.strip()}")
            
            # Get detailed explanation
            explanation = agent.explain_optimization(test_query)
            print(f"\nOptimization Explanation:\n{explanation}")
            
            # Test simple query too
            simple_query = "SELECT COUNT(*) FROM customers WHERE c_mktsegment = 'BUILDING'"
            analysis = agent.analyze_query(simple_query)
            print(f"\nSimple Query Analysis:")
            print(f"LLM Provider: {analysis.get('llm_provider', 'unknown')}")
            print(f"Analysis: {analysis.get('llm_analysis', 'No analysis')}")
        
        logger.info("LLM agent test completed")
        
    except Exception as e:
        logger.error(f"LLM agent test failed: {e}")
        pytest.fail(f"LLM agent test failed: {e}")
        
    finally:
        db.disconnect()
        if os.path.exists("test_enhanced_llm.db"):
            os.remove("test_enhanced_llm.db")

if __name__ == "__main__":
    success = test_enhanced_llm_agent()
    if success:
        print("\nLLM Query Agent is working!")
    else:
        print("\nLLM agent test failed.")