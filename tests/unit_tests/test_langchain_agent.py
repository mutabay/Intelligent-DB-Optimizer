"""
Test LangChain-based query agent.
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.agents.llm_query_agent import LangChainQueryAgent
from src.utils.logging import logger

def test_langchain_agent():
    """Test LangChain query agent."""
    
    logger.info("=== Testing LangChain Query Agent ===")
    
    # Setup
    db = DatabaseSimulator(db_type="sqlite", db_path="test_langchain.db")
    
    try:
        db.connect()
        db.create_sample_tables()
        
        # Add sample data
        db.execute_query("INSERT INTO customers (c_name, c_mktsegment) VALUES ('Customer 1', 'BUILDING')")
        db.execute_query("INSERT INTO orders (o_custkey, o_totalprice) VALUES (1, 1500)")
        
        # Build knowledge graph
        kg = DatabaseSchemaKG(db_type="sqlite")
        kg.build_from_database(db.connection)
        
        # Create LangChain agent
        agent = LangChainQueryAgent(kg, "test_langchain.db")
        
        # Test query
        test_query = "SELECT c.c_name, o.o_totalprice FROM customers c JOIN orders o ON c.c_custkey = o.o_custkey"
        
        print(f"Testing query: {test_query}")
        
        # Analyze query
        analysis = agent.analyze_query(test_query)
        print(f"\nAnalysis: {analysis}")
        
        # Optimize query
        optimization = agent.optimize_query(test_query)
        print(f"\nOptimization: {optimization}")
        
        logger.info("LangChain agent test completed")
        return True
        
    except Exception as e:
        logger.error(f"LangChain agent test failed: {e}")
        return False
        
    finally:
        db.disconnect()
        if os.path.exists("test_langchain.db"):
            os.remove("test_langchain.db")

if __name__ == "__main__":
    success = test_langchain_agent()
    if success:
        print("LangChain Query Agent is working!")
    else:
        print("LangChain agent test failed.")