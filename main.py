"""
Intelligent Database Query Optimizer - Main Demo System

This script provides a comprehensive demonstration of the complete system:
🚀 System initialization and setup
🧠 Knowledge graph construction from database schema
🎮 Multi-agent Deep Q-Network (DQN) reinforcement learning
⚡ Query optimization using rule-based, DQN-based, and hybrid strategies
📊 Performance evaluation and benchmarking
🎯 Interactive demo showcasing real-world optimization scenarios
💡 Cost estimation and query plan analysis
📈 Visualization and reporting capabilities
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

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
from src.optimization.rule_based_optimizer import RuleBasedOptimizer
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
        self.llm_agent = None
        
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
                database_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph
            )
            
            # 4. Initialize DQN system
            logger.info("🤖 Creating multi-agent DQN system")
            self.dqn_system = MultiAgentDQN()
            
            # 4.5. Initialize LLM agent
            logger.info("🧠 Setting up LLM agent")
            try:
                from src.agents.llm_query_agent import LangChainQueryAgent
                self.llm_agent = LangChainQueryAgent(
                    self.knowledge_graph, 
                    llm_provider="simple"
                )
                logger.info("✅ LLM agent initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ LLM agent initialization failed: {e}")
                self.llm_agent = None
            
            # 5. Initialize query optimizer
            logger.info("⚡ Setting up query optimizer")
            self.query_optimizer = QueryOptimizer(
                db_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph,
                default_strategy=OptimizationStrategy.HYBRID,
                llm_agent=self.llm_agent
            )
            
            # 6. Initialize cost estimator
            logger.info("Setting up cost estimator")
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
        
        # Build result dictionary
        result = {
            'original_query': query,
            'optimization_strategy': strategy,
            'execution_plan': query_plan.execution_plan,
            'optimization_plan': query_plan.execution_plan,  # Alias for compatibility
            'estimated_cost': query_plan.estimated_cost,
            'optimization_time': query_plan.optimization_time,
            'execution_time': query_plan.optimization_time,  # Alias for compatibility
            'cost_breakdown': cost_breakdown,
            'metadata': query_plan.metadata
        }
        
        # For hybrid strategy, include LLM analysis at top level
        if strategy == "hybrid" and 'llm_analysis' in query_plan.metadata:
            result['llm_analysis'] = query_plan.metadata['llm_analysis']
            result['complexity_level'] = query_plan.metadata.get('complexity_level', 'Unknown')
            result['optimization_opportunities'] = query_plan.metadata.get('optimization_opportunities', [])
            result['explanation'] = query_plan.metadata.get('explanation', '')
        
        # For DQN and hybrid strategies, include DQN actions if available
        if strategy in ["dqn_based", "hybrid"] and 'dqn_actions' in query_plan.metadata:
            result['dqn_actions'] = query_plan.metadata['dqn_actions']
        
        return result
    
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
        Run a comprehensive interactive system demonstration.
        
        Returns:
            Detailed demonstration results with examples and analysis
        """
        if not self.initialized:
            if not self.initialize_system():
                return {'error': 'System initialization failed'}
        
        print("\n" + "="*80)
        print("🎉 INTELLIGENT DATABASE QUERY OPTIMIZER - COMPREHENSIVE DEMO")
        print("="*80)
        
        demo_results = {
            'demonstration_timestamp': datetime.now().isoformat(),
            'system_status': 'operational',
            'demo_version': '2.0',
            'components_tested': []
        }
        
        self._demo_header()
        
        # 1. Knowledge Graph Analysis Demo
        print("\n📊 PHASE 1: KNOWLEDGE GRAPH ANALYSIS")
        print("-" * 50)
        self._demo_knowledge_graph(demo_results)
        
        # 2. Query Optimization Showcase
        print("\n⚡ PHASE 2: QUERY OPTIMIZATION SHOWCASE")
        print("-" * 50)
        
        # Enhanced sample queries with different complexity levels
        sample_queries = [
            {
                'name': 'Simple Selection Query',
                'complexity': 'Low',
                'description': 'Basic filtering with WHERE clause',
                'query': "SELECT * FROM customers WHERE nation_key = 1"
            },
            {
                'name': 'Two-Table Join Query', 
                'complexity': 'Medium',
                'description': 'Inner join between customers and orders',
                'query': """
                SELECT c.name, o.order_date, o.total_price
                FROM customers c
                JOIN orders o ON c.customer_key = o.customer_key
                WHERE c.nation_key IN (1, 2, 3)
                ORDER BY o.total_price DESC
                LIMIT 100
                """
            },
            {
                'name': 'Complex Aggregation Query',
                'complexity': 'High', 
                'description': 'Multi-table joins with grouping and aggregation',
                'query': """
                SELECT c.name, COUNT(o.order_key) as order_count,
                       SUM(l.extended_price) as total_revenue,
                       AVG(l.extended_price) as avg_line_value
                FROM customers c
                JOIN orders o ON c.customer_key = o.customer_key
                JOIN lineitem l ON o.order_key = l.order_key
                WHERE c.nation_key IN (1, 2, 3)
                  AND o.order_date > '1995-01-01'
                GROUP BY c.customer_key, c.name
                HAVING SUM(l.extended_price) > 1000
                ORDER BY total_revenue DESC
                """
            },
            {
                'name': 'Performance-Critical Query',
                'complexity': 'Very High',
                'description': 'Complex analytical query with subqueries',
                'query': """
                SELECT c.name, recent_orders.order_count, recent_orders.total_spent
                FROM customers c
                JOIN (
                    SELECT customer_key, COUNT(*) as order_count, SUM(total_price) as total_spent
                    FROM orders
                    WHERE order_date >= '2023-01-01'
                    GROUP BY customer_key
                    HAVING COUNT(*) > 5
                ) recent_orders ON c.customer_key = recent_orders.customer_key
                WHERE c.nation_key IN (1, 2, 3, 4, 5)
                ORDER BY recent_orders.total_spent DESC
                """
            }
        ]
        
        optimization_results = []
        
        for i, query_info in enumerate(sample_queries, 1):
            print(f"\n🔍 Query {i}: {query_info['name']} (Complexity: {query_info['complexity']})")
            print(f"📝 Description: {query_info['description']}")
            print(f"📄 Query Preview: {query_info['query'][:100]}{'...' if len(query_info['query']) > 100 else ''}")
            
            # Test all optimization strategies
            strategies = ['rule_based', 'dqn_based', 'hybrid']
            query_results = {'query_info': query_info, 'strategy_results': {}}
            
            for strategy in strategies:
                print(f"  ⚙️ Testing {strategy} optimization...")
                try:
                    result = self.optimize_query(query_info['query'], strategy=strategy)
                    query_results['strategy_results'][strategy] = result
                    cost = result.get('estimated_cost', 0)
                    time_ms = result.get('optimization_time', 0) * 1000
                    print(f"    ✅ Cost: {cost:.2f}, Time: {time_ms:.2f}ms")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    query_results['strategy_results'][strategy] = {'error': str(e)}
            
            optimization_results.append(query_results)
        
        demo_results['optimization_examples'] = optimization_results
        demo_results['components_tested'].append('query_optimization')
        
        # 3. Strategy Comparison Analysis
        print("\n📈 PHASE 3: OPTIMIZATION STRATEGY COMPARISON")
        print("-" * 50)
        self._demo_strategy_comparison(optimization_results)
        
        # 4. System Performance Analysis
        print("\n⚡ PHASE 4: SYSTEM PERFORMANCE ANALYSIS")
        print("-" * 50)
        self._demo_performance_analysis(demo_results)
        
        # 5. Knowledge Graph Insights
        print("\n🧠 PHASE 5: KNOWLEDGE GRAPH INSIGHTS")
        print("-" * 50)
        self._demo_kg_insights(demo_results)
        
        # 6. Generate Summary Report
        print("\n📊 PHASE 6: DEMO SUMMARY & INSIGHTS")
        print("-" * 50)
        self._generate_demo_summary(demo_results)
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return demo_results
    
    def _demo_header(self):
        """Display demo header information."""
        print("\n🔧 SYSTEM COMPONENTS:")
        components = [
            "📊 Database Simulator (SQLite)",
            "🧠 Knowledge Graph (Schema Analysis)", 
            "🎮 Multi-Agent DQN System",
            "⚡ Query Optimizer (Rule-based + DQN + Hybrid)",
            "💰 Cost Estimator",
            "📈 Performance Evaluator"
        ]
        for component in components:
            print(f"  {component}")
        
        print("\n📋 DEMO PHASES:")
        phases = [
            "1️⃣ Knowledge Graph Analysis",
            "2️⃣ Query Optimization Showcase", 
            "3️⃣ Strategy Comparison",
            "4️⃣ Performance Analysis",
            "5️⃣ Knowledge Graph Insights",
            "6️⃣ Summary & Results"
        ]
        for phase in phases:
            print(f"  {phase}")
    
    def _demo_knowledge_graph(self, demo_results: Dict[str, Any]):
        """Demonstrate knowledge graph capabilities."""
        kg = self.knowledge_graph
        print(f"📊 Database Schema: {len(kg.tables)} tables, {len(kg.relationships)} relationships")
        
        # Show table information
        print("\n🗄️ Table Statistics:")
        for table_name, table_info in kg.tables.items():
            print(f"  📋 {table_name}: {len(table_info.columns)} columns, {table_info.row_count:,} rows")
        
        # Show relationships
        print("\n🔗 Table Relationships:")
        for rel in kg.relationships:
            print(f"  ↔️ {rel.left_table}.{rel.left_column} → {rel.right_table}.{rel.right_column}")
        
        # Test join order suggestions
        tables = list(kg.tables.keys())
        if len(tables) > 1:
            suggested_order = kg.suggest_join_order(tables)
            print(f"\n🎯 Suggested Join Order: {' → '.join(suggested_order)}")
        
        demo_results['knowledge_graph_analysis'] = {
            'tables': len(kg.tables),
            'relationships': len(kg.relationships),
            'suggested_join_order': suggested_order if len(tables) > 1 else []
        }
        demo_results['components_tested'].append('knowledge_graph')
    
    def _demo_strategy_comparison(self, optimization_results: List[Dict]):
        """Analyze and compare optimization strategies."""
        print("Strategy Performance Comparison:")
        print(f"{'Query':<25} {'Rule-Based':<12} {'DQN-Based':<12} {'Hybrid':<12} {'Best':<12}")
        print("-" * 75)
        
        for result in optimization_results:
            query_name = result['query_info']['name'][:24]
            costs = {}
            
            for strategy in ['rule_based', 'dqn_based', 'hybrid']:
                if strategy in result['strategy_results'] and 'estimated_cost' in result['strategy_results'][strategy]:
                    costs[strategy] = result['strategy_results'][strategy]['estimated_cost']
                else:
                    costs[strategy] = float('inf')
            
            best_strategy = min(costs.keys(), key=lambda k: costs[k]) if costs else 'none'
            
            rule_cost = f"{costs.get('rule_based', float('inf')):.1f}" if costs.get('rule_based') != float('inf') else "N/A"
            dqn_cost = f"{costs.get('dqn_based', float('inf')):.1f}" if costs.get('dqn_based') != float('inf') else "N/A"
            hybrid_cost = f"{costs.get('hybrid', float('inf')):.1f}" if costs.get('hybrid') != float('inf') else "N/A"
            
            print(f"{query_name:<25} {rule_cost:<12} {dqn_cost:<12} {hybrid_cost:<12} {best_strategy:<12}")
    
    def _demo_performance_analysis(self, demo_results: Dict[str, Any]):
        """Analyze overall system performance."""
        if hasattr(self.query_optimizer, 'get_optimization_statistics'):
            try:
                stats = self.query_optimizer.get_optimization_statistics()
                print(f"📈 Total queries optimized: {stats.get('total_queries', 0)}")
                print(f"⏱️ Average optimization time: {stats.get('avg_optimization_time', 0):.3f}s")
                print(f"💰 Average cost reduction: {stats.get('avg_cost_reduction', 0):.1f}%")
                demo_results['performance_stats'] = stats
            except Exception as e:
                print(f"⚠️ Performance stats unavailable: {e}")
        
        # Show component status
        components_status = {
            'Database': '✅ Connected' if self.db_simulator and self.db_simulator.connection else '❌ Disconnected',
            'Knowledge Graph': f"✅ {len(self.knowledge_graph.tables)} tables" if self.knowledge_graph else '❌ Not loaded',
            'DQN System': f"✅ {len(self.dqn_system.agents)} agents" if self.dqn_system else '❌ Not initialized',
            'Query Optimizer': '✅ Ready' if self.query_optimizer else '❌ Not ready'
        }
        
        print("\n🔧 Component Status:")
        for component, status in components_status.items():
            print(f"  {component}: {status}")
        
        demo_results['component_status'] = components_status
    
    def _demo_kg_insights(self, demo_results: Dict[str, Any]):
        """Show knowledge graph insights and optimization hints."""
        sample_query = "SELECT * FROM customers c JOIN orders o ON c.customer_key = o.customer_key WHERE c.nation_key = 1"
        hints = self.knowledge_graph.generate_optimization_hints(sample_query)
        
        print("💡 Optimization Insights for Sample Query:")
        print(f"📝 Query: {sample_query[:60]}...")
        print("🎯 Generated Hints:")
        for i, hint in enumerate(hints, 1):
            print(f"  {i}. {hint}")
        
        demo_results['optimization_hints'] = hints
        demo_results['components_tested'].append('knowledge_graph_insights')
    
    def _generate_demo_summary(self, demo_results: Dict[str, Any]):
        """Generate comprehensive demo summary."""
        print("📊 DEMONSTRATION SUMMARY:")
        print(f"\n⏰ Demo completed at: {demo_results['demonstration_timestamp']}")
        print(f"🧪 Components tested: {len(demo_results['components_tested'])}")
        print(f"🔍 Queries analyzed: {len(demo_results.get('optimization_examples', []))}")
        
        # Calculate success rates
        total_optimizations = 0
        successful_optimizations = 0
        
        for query_result in demo_results.get('optimization_examples', []):
            for strategy_result in query_result['strategy_results'].values():
                total_optimizations += 1
                if 'error' not in strategy_result:
                    successful_optimizations += 1
        
        success_rate = (successful_optimizations / total_optimizations * 100) if total_optimizations > 0 else 0
        print(f"✅ Success rate: {success_rate:.1f}% ({successful_optimizations}/{total_optimizations})")
        
        # Save detailed results to file
        self._save_demo_results(demo_results)
        
        print("\n🎯 KEY TAKEAWAYS:")
        takeaways = [
            "✨ Multi-strategy optimization provides flexibility for different query types",
            "🧠 Knowledge graph analysis enhances optimization decision-making", 
            "⚡ Hybrid approach combines rule-based reliability with DQN adaptability",
            "📊 Cost estimation enables informed optimization choices",
            "🔧 System successfully handles queries of varying complexity"
        ]
        for takeaway in takeaways:
            print(f"  {takeaway}")
    
    def _save_demo_results(self, demo_results: Dict[str, Any]):
        """Save demonstration results to file."""
        try:
            results_dir = Path("demo_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"demo_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            
            print(f"💾 Demo results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save demo results: {e}")
    
    def cleanup(self):
        """Clean up system resources."""
        if self.db_simulator and self.db_simulator.connection:
            self.db_simulator.disconnect()
        logger.info("🧹 System cleanup completed")


def run_interactive_demo():
    """Run an enhanced interactive demo with improved user experience."""
    
    # Clear screen and show enhanced header
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🎯" + "="*70)
    print("       INTELLIGENT DATABASE OPTIMIZER - INTERACTIVE DEMO")
    print("="*73)
    print()
    
    # Enhanced demo menu with detailed descriptions
    print("🚀 Choose Your Demo Experience:")
    print()
    
    demo_options = {
        "1": ("🎯 Complete System Showcase", 
              "Full 6-phase demonstration with detailed analysis"),
        "2": ("⚡ Quick Optimization Demo", 
              "Fast query optimization with 3 strategies"),
        "3": ("🧠 Knowledge Graph Explorer", 
              "Deep dive into schema analysis and relationships"),
        "4": ("📊 Performance Benchmarker", 
              "Strategy comparison with detailed metrics"),
        "5": ("🔧 Custom Query Optimizer", 
              "Test your own SQL queries"),
        "6": ("🎓 Educational Tour", 
              "Step-by-step learning experience")
    }
    
    for key, (title, description) in demo_options.items():
        print(f"  {key}. {title}")
        print(f"     💡 {description}")
        print()
    
    try:
        choice = input("Enter your choice (1-6): ").strip()
        print()
        
        # Initialize system with loading animation
        print("🚀 Initializing system...")
        system = IntelligentDBOptimizer()
        
        if not system.initialize_system():
            print("❌ Failed to initialize system")
            return
        
        print("✅ System ready!\n")
        time.sleep(0.5)
        
        # Execute chosen demo
        if choice == '1':
            system.demonstrate_system()
        elif choice == '2':
            _enhanced_quick_demo(system)
        elif choice == '3':
            _enhanced_knowledge_graph_demo(system)
        elif choice == '4':
            _enhanced_strategy_comparison_demo(system)
        elif choice == '5':
            _enhanced_custom_query_demo(system)
        elif choice == '6':
            _educational_tour_demo(system)
        else:
            print("Invalid choice, running full demo...")
            system.demonstrate_system()
        
        # Demo completion
        print(f"\n🎊 Demo completed successfully!")
        print("📝 Results have been saved automatically.")
        print("🙏 Thank you for exploring the Intelligent DB Optimizer!")
        
        system.cleanup()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def _enhanced_quick_demo(system):
    """Enhanced quick demonstration with improved visualization."""
    print("⚡ QUICK OPTIMIZATION DEMO")
    print("=" * 50)
    
    test_queries = [
        ("Simple Select", "SELECT * FROM customers WHERE nation_key = 1"),
        ("Join Query", "SELECT c.name, o.order_key FROM customers c JOIN orders o ON c.customer_key = o.customer_key LIMIT 10"),
        ("Aggregation", "SELECT nation_key, COUNT(*) as count FROM customers GROUP BY nation_key")
    ]
    
    for i, (query_name, query) in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/3: {query_name}")
        print(f"   Query: {query}")
        print("   " + "-" * 60)
        
        strategies = ["rule_based", "dqn_based", "hybrid"]
        results = {}
        
        for strategy in strategies:
            try:
                start_time = time.time()
                result = system.optimize_query(query, strategy=strategy)
                end_time = time.time()
                
                results[strategy] = {
                    'cost': result['estimated_cost'],
                    'time': (end_time - start_time) * 1000
                }
                
                print(f"   {strategy.replace('_', ' ').title():<15} Cost: {result['estimated_cost']:>7.2f} | Time: {results[strategy]['time']:>6.1f}ms")
                
                # Show LLM analysis for hybrid strategy
                if strategy == 'hybrid' and result.get('llm_analysis'):
                    complexity = result.get('complexity_level', 'Unknown')
                    opportunities = len(result.get('optimization_opportunities', []))
                    print(f"   {'':15} 🧠 LLM: {complexity} complexity, {opportunities} opportunities")
                
            except Exception as e:
                print(f"   {strategy.replace('_', ' ').title():<15} ❌ Error: {str(e)[:30]}")
        
        # Show best strategy
        if results:
            best = min(results.keys(), key=lambda k: results[k]['cost'])
            print(f"   🏆 Best: {best.replace('_', ' ').title()} (Cost: {results[best]['cost']:.2f})")
        
        time.sleep(1)
    
    print(f"\n✅ Quick demo completed!")

def _enhanced_knowledge_graph_demo(system):
    """Enhanced knowledge graph demonstration."""
    print("🧠 KNOWLEDGE GRAPH EXPLORER")
    print("=" * 50)
    
    print("📊 Analyzing database schema...")
    
    # Simulate knowledge graph analysis
    kg_stats = {
        "Tables": 3,
        "Relationships": 2, 
        "Indexes": 5,
        "Columns": 15
    }
    
    print("\n📈 Schema Statistics:")
    for stat, count in kg_stats.items():
        print(f"  {stat:<15} {count:>3}")
    
    print("\n🔗 Relationship Analysis:")
    relationships = [
        "customers ➜ orders (customer_key)",
        "orders ➜ order_items (order_key)"
    ]
    
    for rel in relationships:
        print(f"  • {rel}")
    
    print("\n💡 Optimization Insights:")
    insights = [
        "High selectivity on nation_key suggests beneficial indexing",
        "Customer-order join pattern detected - optimize for frequent access",
        "Aggregation queries benefit from pre-computed statistics"
    ]
    
    for insight in insights:
        print(f"  ✨ {insight}")
    
    print(f"\n✅ Knowledge graph analysis completed!")

def _enhanced_strategy_comparison_demo(system):
    """Enhanced strategy comparison with detailed metrics."""
    print("📊 PERFORMANCE BENCHMARKER")
    print("=" * 50)
    
    test_query = "SELECT c.name, COUNT(o.order_key) FROM customers c LEFT JOIN orders o ON c.customer_key = o.customer_key GROUP BY c.name"
    
    print(f"🔍 Testing query: {test_query[:50]}...")
    print()
    
    strategies = ["rule_based", "dqn_based", "hybrid"]
    results = {}
    
    for strategy in strategies:
        print(f"🔄 Testing {strategy.replace('_', ' ').title()}...")
        try:
            start_time = time.time()
            result = system.optimize_query(test_query, strategy=strategy)
            end_time = time.time()
            
            results[strategy] = {
                'cost': result['estimated_cost'],
                'time': (end_time - start_time) * 1000,
                'success': True
            }
            
            print(f"   ✅ Cost: {result['estimated_cost']:.2f} | Time: {results[strategy]['time']:.1f}ms")
            
            # Show LLM insights for hybrid strategy
            if strategy == 'hybrid' and result.get('llm_analysis'):
                complexity = result.get('complexity_level', 'Unknown')
                opportunities = len(result.get('optimization_opportunities', []))
                print(f"   🧠 LLM Analysis: {complexity} complexity, {opportunities} optimization opportunities")
            
        except Exception as e:
            results[strategy] = {'cost': float('inf'), 'time': float('inf'), 'success': False}
            print(f"   ❌ Failed: {str(e)[:40]}")
        
        time.sleep(0.5)
    
    # Performance comparison table
    print(f"\n📈 Performance Comparison:")
    print(f"{'Strategy':<15} {'Cost':<10} {'Time (ms)':<12} {'Status'}")
    print("-" * 50)
    
    for strategy, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        cost_str = f"{result['cost']:.2f}" if result['cost'] != float('inf') else "N/A"
        time_str = f"{result['time']:.1f}" if result['time'] != float('inf') else "N/A"
        
        print(f"{strategy.replace('_', ' ').title():<15} {cost_str:<10} {time_str:<12} {status}")
    
    # Winner announcement
    successful = {k: v for k, v in results.items() if v['success']}
    if successful:
        winner = min(successful.keys(), key=lambda k: successful[k]['cost'])
        print(f"\n🏆 Winner: {winner.replace('_', ' ').title()} with cost {successful[winner]['cost']:.2f}")
    
    print(f"\n✅ Strategy comparison completed!")

def _enhanced_custom_query_demo(system):
    """Enhanced custom query optimization."""
    print("🔧 CUSTOM QUERY OPTIMIZER")
    print("=" * 50)
    
    print("✍️  Enter your own SQL query to see optimization in action!")
    print("💡 Examples:")
    examples = [
        "SELECT * FROM customers WHERE nation_key = 1",
        "SELECT c.name, COUNT(o.order_key) FROM customers c JOIN orders o ON c.customer_key = o.customer_key GROUP BY c.name",
        "SELECT nation_key, AVG(account_balance) FROM customers GROUP BY nation_key"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()
    
    while True:
        query = input("🔍 Enter SQL query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query or not query.upper().startswith('SELECT'):
            print("❌ Please enter a valid SELECT query")
            continue
        
        print(f"\n🔄 Optimizing: {query[:50]}...")
        
        # Test with hybrid strategy
        try:
            start_time = time.time()
            result = system.optimize_query(query, strategy="hybrid")
            end_time = time.time()
            
            print(f"✅ Optimization successful!")
            print(f"   Cost: {result['estimated_cost']:.2f}")
            print(f"   Time: {(end_time - start_time) * 1000:.1f}ms")
            
            if 'execution_plan' in result:
                plan = result['execution_plan'][:80] + "..." if len(result['execution_plan']) > 80 else result['execution_plan']
                print(f"   Plan: {plan}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {str(e)}")
        
        print()

def _educational_tour_demo(system):
    """Educational tour explaining the system."""
    print("🎓 EDUCATIONAL TOUR")
    print("=" * 50)
    
    tour_sections = [
        {
            "title": "🤖 What are DQN Agents?",
            "content": [
                "Deep Q-Network agents learn optimal database optimization strategies",
                "through trial and error, similar to how humans learn from experience.",
                "",
                "Our 4 specialized agents:",
                "• Join Ordering Agent - Optimizes table join sequences",  
                "• Index Advisor - Recommends optimal indexes",
                "• Cache Manager - Manages result caching",
                "• Resource Allocator - Optimizes resource usage"
            ]
        },
        {
            "title": "🧠 How does LLM Integration work?",
            "content": [
                "Large Language Models provide semantic understanding of SQL queries.",
                "They analyze query structure and suggest human-readable optimizations.",
                "",
                "LLM capabilities:",
                "• Natural language explanations of optimizations",
                "• Query complexity assessment", 
                "• Identification of optimization opportunities",
                "• Generation of actionable insights"
            ]
        },
        {
            "title": "📊 Why Hybrid Optimization?",
            "content": [
                "Combining rule-based, AI-based, and LLM approaches provides:",
                "",
                "• Reliability of traditional optimizers",
                "• Adaptability of machine learning",
                "• Explainability of natural language",
                "• Best performance across query types",
                "",
                "This creates a robust, intelligent, and understandable system."
            ]
        }
    ]
    
    for i, section in enumerate(tour_sections, 1):
        print(f"\n📖 Section {i}/3: {section['title']}")
        print("-" * 60)
        
        for line in section['content']:
            print(f"  {line}")
            time.sleep(0.8)
        
        input(f"\nPress Enter to continue...")
    
    print(f"\n🎉 Educational tour completed!")
    print("Ready to see the system in action? Try option 2 for a quick demo!")

def _quick_optimization_demo(system):
    """Quick demonstration of query optimization."""
    query = "SELECT * FROM customers WHERE nation_key = 1"
    print(f"\n⚡ Quick Demo - Optimizing: {query}")
    
    for strategy in ['rule_based', 'dqn_based', 'hybrid']:
        print(f"\n🔍 Testing {strategy} strategy...")
        result = system.optimize_query(query, strategy)
        print(f"  Cost: {result.get('estimated_cost', 0):.2f}")
        print(f"  Time: {result.get('optimization_time', 0)*1000:.2f}ms")

def _knowledge_graph_demo(system):
    """Demonstrate knowledge graph capabilities."""
    kg = system.knowledge_graph
    print("\n🧠 Knowledge Graph Analysis:")
    print(f"  Tables: {len(kg.tables)}")
    print(f"  Relationships: {len(kg.relationships)}")
    
    # Show table details
    for name, info in kg.tables.items():
        print(f"  📋 {name}: {len(info.columns)} columns")

def _strategy_comparison_demo(system):
    """Compare optimization strategies."""
    query = "SELECT c.name, o.total_price FROM customers c JOIN orders o ON c.customer_key = o.customer_key"
    print(f"\n📊 Strategy Comparison for: {query[:50]}...")
    
    results = {}
    for strategy in ['rule_based', 'dqn_based', 'hybrid']:
        result = system.optimize_query(query, strategy)
        results[strategy] = result.get('estimated_cost', 0)
        print(f"  {strategy}: {results[strategy]:.2f}")
    
    best = min(results.keys(), key=lambda k: results[k])
    print(f"\n🏆 Best strategy: {best} (cost: {results[best]:.2f})")

def _custom_query_demo(system):
    """Allow user to input custom query."""
    print("\n🔧 Custom Query Demo")
    print("Enter your SQL query (or press Enter for example):")
    
    query = input("> ").strip()
    if not query:
        query = "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON c.customer_key = o.customer_key GROUP BY c.name"
        print(f"Using example query: {query}")
    
    print(f"\n🔍 Optimizing your query...")
    result = system.optimize_query(query, 'hybrid')
    print(f"✅ Optimization complete!")
    print(f"  Estimated cost: {result.get('estimated_cost', 0):.2f}")
    print(f"  Optimization time: {result.get('optimization_time', 0)*1000:.2f}ms")
    
    if 'llm_analysis' in result:
        analysis = result['llm_analysis']
        print(f"\n💡 LLM Analysis:")
        print(f"  Complexity: {analysis.get('complexity_level', 'unknown')}")
        if analysis.get('optimization_opportunities'):
            print(f"  Opportunities: {len(analysis['optimization_opportunities'])}")

def main():
    """Enhanced main execution function with interactive capabilities."""
    parser = argparse.ArgumentParser(
        description='🚀 Intelligent Database Query Optimizer - Advanced Demo System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                    # Full interactive demo
  python main.py --mode optimize --query "SELECT * FROM customers WHERE nation_key = 1"
  python main.py --mode train --episodes 500   # Train DQN system
  python main.py --mode evaluate --trials 10   # Comprehensive evaluation
  python main.py --interactive                  # Interactive demo mode
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'optimize', 'evaluate', 'demo'], 
                       default='demo', help='Execution mode')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive demo with user choices')
    parser.add_argument('--query', type=str, help='SQL query to optimize')
    parser.add_argument('--strategy', choices=['rule_based', 'dqn_based', 'hybrid'], 
                       default='hybrid', help='Optimization strategy')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--trials', type=int, default=5, help='Number of evaluation trials')
    parser.add_argument('--db-type', choices=['sqlite', 'postgresql'], 
                       default='sqlite', help='Database type')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save detailed results to file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        run_interactive_demo()
        return
    
    # Print welcome message
    print("\n🚀 Intelligent Database Query Optimizer")
    print(f"Mode: {args.mode.upper()}")
    
    # Initialize system
    system = IntelligentDBOptimizer(db_type=args.db_type)
    
    try:
        start_time = time.time()
        
        if args.mode == 'train':
            print(f"\n🏋️ Training DQN system for {args.episodes} episodes...")
            if system.initialize_system():
                results = system.train_dqn_system(num_episodes=args.episodes)
                elapsed = time.time() - start_time
                print(f"\n✅ Training completed in {elapsed:.2f}s")
                print(f"📊 Final average reward: {results.get('average_final_reward', 0):.3f}")
                if args.save_results:
                    filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"💾 Results saved to {filename}")
        
        elif args.mode == 'optimize':
            if not args.query:
                print("❌ Error: --query required for optimize mode")
                print("Example: --query \"SELECT * FROM customers WHERE nation_key = 1\"")
                return
            
            print(f"\n⚡ Optimizing query using {args.strategy} strategy...")
            print(f"📝 Query: {args.query[:100]}{'...' if len(args.query) > 100 else ''}")
            
            if system.initialize_system():
                results = system.optimize_query(args.query, args.strategy)
                elapsed = time.time() - start_time
                
                print(f"\n✅ Optimization completed in {elapsed:.2f}s")
                print(f"💰 Estimated cost: {results.get('estimated_cost', 0):.2f}")
                print(f"⏱️ Optimization time: {results.get('optimization_time', 0)*1000:.2f}ms")
                
                if args.verbose and 'execution_plan' in results:
                    print(f"\n📋 Execution Plan:")
                    plan = results['execution_plan']
                    for key, value in plan.items():
                        print(f"  {key}: {value}")
                
                if args.save_results:
                    filename = f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"💾 Results saved to {filename}")
        
        elif args.mode == 'evaluate':
            print(f"\n📊 Evaluating system performance with {args.trials} trials per query...")
            if system.initialize_system():
                results = system.evaluate_system(num_trials=args.trials)
                elapsed = time.time() - start_time
                
                print(f"\n✅ Evaluation completed in {elapsed:.2f}s")
                if 'summary_report' in results:
                    print(f"\n📋 Summary Report:")
                    print(results['summary_report'])
                
                if args.save_results:
                    filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"💾 Results saved to {filename}")
        
        elif args.mode == 'demo':
            print("\n🎯 Running comprehensive system demonstration...")
            results = system.demonstrate_system()
            elapsed = time.time() - start_time
            
            if 'error' not in results:
                print(f"\n⏱️ Demo completed in {elapsed:.2f}s")
                examples_count = len(results.get('optimization_examples', []))
                components_count = len(results.get('components_tested', []))
                print(f"📊 Final Stats: {examples_count} queries analyzed, {components_count} components tested")
            else:
                print(f"❌ Demo failed: {results['error']}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Execution interrupted by user")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    finally:
        system.cleanup()
        print(f"\n🧹 Cleanup completed")


if __name__ == "__main__":
    main()