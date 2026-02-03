"""
Main Execution Script for Intelligent Database Query Optimizer

This script demonstrates the complete system functionality including:
- Database setup and initialization
- Knowledge graph construction  
- Multi-agent DQN system training
- Query optimization using different strategies
- Performance evaluation against baselines
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.agents.rl_environment import QueryOptimizationEnv
from src.agents.dqn_agent import MultiAgentDQN
from src.agents.dqn_trainer import DQNTrainer
from src.agents.dqn_evaluator import DQNEvaluator, QueryBenchmark
from src.optimization.query_optimizer import QueryOptimizer, OptimizationStrategy
from src.optimization.cost_estimator import CostEstimator
from evaluation.baselines.rule_based_optimizer import RuleBasedOptimizer
from src.utils.logging import logger


class IntelligentDBOptimizer:
    """
    Main system controller for the Intelligent Database Query Optimizer.
    
    This class coordinates all system components and provides high-level
    interfaces for training, optimization, and evaluation.
    """
    
    def __init__(self, db_type: str = "sqlite", db_path: str = ":memory:"):
        """
        Initialize the intelligent database optimizer system.
        
        Args:
            db_type: Type of database ("sqlite" or "postgresql")
            db_path: Database connection path
        """
        self.db_type = db_type
        self.db_path = db_path
        
        # Initialize core components
        self.db_simulator = None
        self.knowledge_graph = None
        self.rl_environment = None
        self.dqn_system = None
        self.query_optimizer = None
        self.cost_estimator = None
        
        # Training and evaluation components
        self.trainer = None
        self.evaluator = None
        self.baseline_optimizer = None
        
        # System status
        self.initialized = False
        
    def initialize_system(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing Intelligent Database Optimizer System")
            
            # 1. Initialize database simulator
            logger.info("📊 Setting up database simulator")
            self.db_simulator = DatabaseSimulator(db_type=self.db_type, db_path=self.db_path)
            self.db_simulator.connect()
            self.db_simulator.create_sample_tables()
            
            # 2. Build knowledge graph
            logger.info("🧠 Building database knowledge graph")
            self.knowledge_graph = DatabaseSchemaKG(db_type=self.db_type)
            self.knowledge_graph.build_from_database(self.db_simulator.connection)
            
            # 3. Initialize RL environment
            logger.info("🎮 Setting up RL environment")
            self.rl_environment = QueryOptimizationEnv(
                db_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph
            )
            
            # 4. Initialize DQN system
            logger.info("🤖 Creating multi-agent DQN system")
            self.dqn_system = MultiAgentDQN()
            
            # 5. Initialize query optimizer
            logger.info("⚡ Setting up query optimizer")
            self.query_optimizer = QueryOptimizer(
                db_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph,
                default_strategy=OptimizationStrategy.HYBRID
            )
            
            # 6. Initialize cost estimator
            logger.info("💰 Setting up cost estimator")
            self.cost_estimator = CostEstimator()
            self._register_table_statistics()
            
            # 7. Initialize training and evaluation components
            logger.info("🏋️ Setting up training infrastructure")
            self.trainer = DQNTrainer()
            self.evaluator = DQNEvaluator()
            self.baseline_optimizer = RuleBasedOptimizer()
            
            self.initialized = True
            logger.info("✅ System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def _register_table_statistics(self):
        """Register table statistics with the cost estimator."""
        # Sample statistics for TPC-H-like schema
        table_stats = {
            'customers': {
                'row_count': 1000,
                'page_count': 50,
                'average_row_size': 150,
                'column_statistics': {
                    'customer_key': {'cardinality': 1000, 'null_fraction': 0.0},
                    'nation_key': {'cardinality': 25, 'null_fraction': 0.0}
                }
            },
            'orders': {
                'row_count': 5000,
                'page_count': 200,
                'average_row_size': 120,
                'column_statistics': {
                    'order_key': {'cardinality': 5000, 'null_fraction': 0.0},
                    'customer_key': {'cardinality': 1000, 'null_fraction': 0.0}
                }
            },
            'lineitem': {
                'row_count': 20000,
                'page_count': 800,
                'average_row_size': 100,
                'column_statistics': {
                    'order_key': {'cardinality': 5000, 'null_fraction': 0.0},
                    'line_number': {'cardinality': 7, 'null_fraction': 0.0}
                }
            }
        }
        
        for table_name, stats in table_stats.items():
            self.cost_estimator.register_table_statistics(table_name, stats)
    
    def train_dqn_system(self, num_episodes: int = 1000, save_interval: int = 100) -> Dict[str, Any]:
        """
        Train the DQN system using reinforcement learning.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Save models every N episodes
            
        Returns:
            Training results and statistics
        """
        if not self.initialized:
            logger.error("System not initialized. Call initialize_system() first.")
            return {}
        
        logger.info(f"🏋️ Starting DQN training for {num_episodes} episodes")
        
        # Train the DQN system
        self.trainer.train(
            env=self.rl_environment,
            dqn=self.dqn_system,
            num_episodes=num_episodes,
            save_every=save_interval
        )
        
        # Generate training plots
        self.trainer.plot_training_progress()
        
        # Return training statistics
        return {
            'episodes_trained': num_episodes,
            'final_rewards': self.trainer.episode_rewards[-10:] if self.trainer.episode_rewards else [],
            'average_final_reward': sum(self.trainer.episode_rewards[-10:]) / 10 if len(self.trainer.episode_rewards) >= 10 else 0,
            'total_training_time': time.time()
        }
    
    def optimize_query(self, query: str, strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Optimize a query using the specified strategy.
        
        Args:
            query: SQL query to optimize
            strategy: Optimization strategy ("rule_based", "dqn_based", "hybrid")
            
        Returns:
            Optimization results
        """
        if not self.initialized:
            logger.error("System not initialized. Call initialize_system() first.")
            return {}
        
        # Map string to enum
        strategy_map = {
            'rule_based': OptimizationStrategy.RULE_BASED,
            'dqn_based': OptimizationStrategy.DQN_BASED,
            'hybrid': OptimizationStrategy.HYBRID
        }
        
        optimization_strategy = strategy_map.get(strategy, OptimizationStrategy.HYBRID)
        
        logger.info(f"⚡ Optimizing query using {strategy} strategy")
        
        # Optimize the query
        query_plan = self.query_optimizer.optimize_query(query, optimization_strategy)
        
        # Get cost breakdown
        cost_breakdown = self.cost_estimator.get_cost_breakdown(query_plan.execution_plan)
        
        return {
            'original_query': query,
            'optimization_strategy': strategy,
            'execution_plan': query_plan.execution_plan,
            'estimated_cost': query_plan.estimated_cost,
            'optimization_time': query_plan.optimization_time,
            'cost_breakdown': cost_breakdown,
            'metadata': query_plan.metadata
        }
    
    def evaluate_system(self, num_trials: int = 5) -> Dict[str, Any]:
        """
        Evaluate the DQN system against baseline optimizers.
        
        Args:
            num_trials: Number of evaluation trials per query
            
        Returns:
            Comprehensive evaluation results
        """
        if not self.initialized:
            logger.error("System not initialized. Call initialize_system() first.")
            return {}
        
        logger.info(f"📊 Starting system evaluation with {num_trials} trials per query")
        
        # Create query benchmark
        benchmark = QueryBenchmark(self.db_simulator)
        
        # Run evaluation
        evaluation_results = self.evaluator.evaluate_dqn_vs_baseline(
            env=self.rl_environment,
            dqn=self.dqn_system,
            baseline_optimizer=self.baseline_optimizer,
            benchmark=benchmark,
            num_trials=num_trials
        )
        
        # Generate evaluation plots
        self.evaluator.plot_evaluation_results(evaluation_results)
        
        # Generate text report
        report = self.evaluator.generate_evaluation_report(evaluation_results)
        
        return {
            'evaluation_results': evaluation_results,
            'summary_report': report,
            'total_queries_evaluated': len(benchmark.benchmark_queries),
            'evaluation_timestamp': evaluation_results['timestamp']
        }
    
    def demonstrate_system(self) -> Dict[str, Any]:
        """
        Run a complete system demonstration.
        
        Returns:
            Demonstration results
        """
        if not self.initialized:
            if not self.initialize_system():
                return {'error': 'System initialization failed'}
        
        demo_results = {
            'demonstration_timestamp': time.time(),
            'system_status': 'operational'
        }
        
        # 1. Demonstrate query optimization
        logger.info("🎯 Demonstrating query optimization")
        sample_queries = [
            "SELECT * FROM customers WHERE nation_key = 1",
            """
            SELECT c.name, o.order_date, o.total_price
            FROM customers c
            JOIN orders o ON c.customer_key = o.customer_key
            WHERE c.nation_key = 1
            """,
            """
            SELECT c.name, SUM(o.total_price) as total_spent
            FROM customers c
            JOIN orders o ON c.customer_key = o.customer_key
            GROUP BY c.customer_key, c.name
            HAVING SUM(o.total_price) > 10000
            """
        ]
        
        optimization_results = []
        for i, query in enumerate(sample_queries):
            logger.info(f"Optimizing sample query {i+1}")
            result = self.optimize_query(query, strategy="hybrid")
            optimization_results.append(result)
        
        demo_results['optimization_examples'] = optimization_results
        
        # 2. Show system statistics
        optimizer_stats = self.query_optimizer.get_optimization_statistics()
        demo_results['system_statistics'] = optimizer_stats
        
        logger.info("✅ System demonstration completed")
        return demo_results
    
    def cleanup(self):
        """Clean up system resources."""
        if self.db_simulator and self.db_simulator.connection:
            self.db_simulator.disconnect()
        logger.info("🧹 System cleanup completed")


def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description='Intelligent Database Query Optimizer')
    parser.add_argument('--mode', choices=['train', 'optimize', 'evaluate', 'demo'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--query', type=str, help='SQL query to optimize')
    parser.add_argument('--strategy', choices=['rule_based', 'dqn_based', 'hybrid'], 
                       default='hybrid', help='Optimization strategy')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--trials', type=int, default=5, help='Number of evaluation trials')
    parser.add_argument('--db-type', choices=['sqlite', 'postgresql'], 
                       default='sqlite', help='Database type')
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntelligentDBOptimizer(db_type=args.db_type)
    
    try:
        if args.mode == 'train':
            if system.initialize_system():
                results = system.train_dqn_system(num_episodes=args.episodes)
                print(f"Training completed: {results}")
        
        elif args.mode == 'optimize':
            if not args.query:
                print("Error: --query required for optimize mode")
                return
            if system.initialize_system():
                results = system.optimize_query(args.query, args.strategy)
                print(f"Optimization results: {results}")
        
        elif args.mode == 'evaluate':
            if system.initialize_system():
                results = system.evaluate_system(num_trials=args.trials)
                print(f"Evaluation completed: {results['summary_report']}")
        
        elif args.mode == 'demo':
            results = system.demonstrate_system()
            print("🎉 Demonstration completed successfully!")
            print(f"Optimized {len(results.get('optimization_examples', []))} sample queries")
            print(f"System statistics: {results.get('system_statistics', {})}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"Error: {e}")
    
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()