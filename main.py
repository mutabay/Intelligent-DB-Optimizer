#!/usr/bin/env python3
"""
Intelligent Database Query Optimizer

A hybrid AI system for intelligent database query optimization integrating:
- Symbolic AI (rule-based heuristics + knowledge graph)
- Reinforcement Learning (DQN-based optimization)
- Generative AI (LLM semantic analysis)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.optimization import HybridOptimizer, OptimizationResult
from src.utils.logging import logger


class Application:
    """Main application controller."""
    
    def __init__(self, db_type: str = "sqlite", db_path: str = ":memory:"):
        self.db_type = db_type
        self.db_path = db_path
        self.db_simulator = None
        self.knowledge_graph = None
        self.optimizer = None
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            # Database
            self.db_simulator = DatabaseSimulator(db_type=self.db_type, db_path=self.db_path)
            self.db_simulator.connect()
            self.db_simulator.create_sample_tables()
            logger.info(f"Database initialized ({self.db_type})")
            
            # Knowledge Graph
            self.knowledge_graph = DatabaseSchemaKG()
            self.knowledge_graph.build_from_database(self.db_simulator.connection)
            logger.info(f"Knowledge graph built ({len(self.knowledge_graph.tables)} tables)")
            
            # Hybrid Optimizer
            self.optimizer = HybridOptimizer(self.db_simulator, self.knowledge_graph)
            logger.info("Hybrid optimizer ready")
            
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def optimize(self, query: str) -> OptimizationResult:
        """Optimize a SQL query."""
        return self.optimizer.optimize(query)
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_simulator:
            self.db_simulator.disconnect()


def demo(app: Application):
    """Run demonstration."""
    print("\n" + "="*60)
    print("  INTELLIGENT DATABASE QUERY OPTIMIZER")
    print("  Hybrid AI Optimization Demo")
    print("="*60)
    
    queries = [
        "SELECT * FROM customers WHERE c_custkey = 1",
        "SELECT c.c_name, o.o_totalprice FROM customers c JOIN orders o ON c.c_custkey = o.o_custkey",
        "SELECT c.c_name, COUNT(*) FROM customers c JOIN orders o ON c.c_custkey = o.o_custkey GROUP BY c.c_name"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query[:50]}...")
        result = app.optimize(query)
        print(f"  Cost: {result.estimated_cost:.1f}")
        print(f"  Time: {result.optimization_time*1000:.1f}ms")
        print(f"  Strategy: {result.strategy}")
        print(f"  Confidence: {result.confidence:.2f}")
    
    print("\n" + "="*60)
    print("Demo completed.")


def interactive(app: Application):
    """Run interactive mode."""
    print("\n" + "="*60)
    print("  INTELLIGENT DATABASE QUERY OPTIMIZER")
    print("  Interactive Mode")
    print("="*60)
    print("\nEnter SQL queries to optimize (type 'exit' to quit)\n")
    
    while True:
        try:
            query = input("SQL> ").strip()
            if query.lower() in ('exit', 'quit', 'q'):
                break
            if not query:
                continue
            
            result = app.optimize(query)
            print(f"\nResults:")
            print(f"  Estimated Cost: {result.estimated_cost:.1f}")
            print(f"  Optimization Time: {result.optimization_time*1000:.1f}ms")
            print(f"  Strategy: {result.strategy}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Explanation: {result.explanation}")
            print()
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def optimize_single(app: Application, query: str, verbose: bool = False):
    """Optimize a single query."""
    result = app.optimize(query)
    
    print(f"\nQuery: {result.query}")
    print(f"Estimated Cost: {result.estimated_cost:.1f}")
    print(f"Optimization Time: {result.optimization_time*1000:.1f}ms")
    print(f"Strategy: {result.strategy}")
    print(f"Confidence: {result.confidence:.2f}")
    
    if verbose:
        print(f"Explanation: {result.explanation}")
        print(f"Execution Plan: {result.execution_plan}")
        print(f"Active Tiers: {result.metadata.get('tiers_used', [])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Intelligent Database Query Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                           Run demonstration
  python main.py --interactive                    Interactive mode
  python main.py --query "SELECT * FROM users"    Optimize single query
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--query', type=str, help='SQL query to optimize')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize application
    app = Application()
    if not app.initialize():
        print("Failed to initialize system")
        return 1
    
    try:
        if args.demo:
            demo(app)
        elif args.interactive:
            interactive(app)
        elif args.query:
            optimize_single(app, args.query, args.verbose)
        else:
            parser.print_help()
    finally:
        app.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
