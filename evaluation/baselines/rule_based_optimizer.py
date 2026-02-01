""" 
Rule-based baseline optimizer for comparison with AI system.
"""
import sqlite3
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.database_environment import db_simulator
from src.database_environment.db_simulator import DatabaseSimulator, QueryResult
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.utils.logging import logger

@dataclass
class OptimizationResult:
    """Result of query optimization."""
    original_query: str
    optimized_query: str
    optimization_strategy: str
    estimated_improvement: float
    execution_result: Optional[QueryResult] = None

class RuleBasedOptimizer:
    """
    Simple rule-based optimizer that usees traditional heuristics.
    This serves as our baseline for comparison. 
    """

    def __init__(self, knowledge_graph: DatabaseSchemaKG):
        self.kg = knowledge_graph
        self.optimization_rules = [
            "smallest_table_first",
            "push_predicates_down",
            "use_indexes_for_joins",
            "avoid_cartesian_products"
        ]
        
    def optimize_query(self, query: str) -> OptimizationResult:
        """
        Optimize the given SQL query using rule-based strategies.
        
        Args:
            query: Original SQL query
            
        Returns:
            OptimizationResult with details of optimization
        """
        logger.info(f"Optimizing query with rule-based approach")
        
        # Parse query to extract components
        query_components = self._parse_query_components(query)

        # Apply optimization rules
        optimized_query = query  # start with original query
        strategies_applied = []
        estimated_improvement = 0.0

        # Rule 1: Reorder joins based on table sizes
        if len(query_components['tables']) > 1:
            optimal_order = self.kg.suggest_join_order(query_components['tables'])
            if optimal_order != query_components['tables']:
                optimized_query = self._reorder_joins(query, optimal_order)
                strategies_applied.append("join_reordering")
                estimated_improvement += 0.15  # Estimate 15% improvement

        # Rule 2: Add join hints for foreign key relationships
        if self._has_foreign_key_joins(query_components):
            optimized_query = self._add_join_hints(optimized_query, query_components)
            strategies_applied.append("foreign_key_hints")
            estimated_improvement += 0.10  # Estimate 10% improvement

        # Rule 3: Suggest index usage
        if query_components['where_conditions']:
            optimized_query = self._add_index_hints(optimized_query, query_components)
            strategies_applied.append("index_suggestions")
            estimated_improvement += 0.20  # Estimate 20% improvement

        strategy_description = ", ".join(strategies_applied) if strategies_applied else "no_optimization"
        
        return OptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            optimization_strategy=strategy_description,
            estimated_improvement=estimated_improvement
        )
    
    def _parse_query_components(self, query: str) -> Dict[str, List[str]]:
        """
        Simple query parsing to extract tables, joins, and conditions.
        This is a simplified parser - in production, use a proper SQL parser.
        """
        query_upper = query.upper()
        
        # Extract table names (simple approach)
        tables = []
        for table_name in self.kg.tables.keys():
            if table_name.upper() in query_upper:
                tables.append(table_name)
        
        # Extract WHERE conditions (simple approach)
        where_conditions = []
        if 'WHERE' in query_upper:
            # This is very simplified - just look for column references
            for table_name, table_info in self.kg.tables.items():
                for column in table_info.columns:
                    if f"{column.upper()}" in query_upper:
                        where_conditions.append(f"{table_name}.{column}")
        
        # Extract JOIN information
        joins = []
        if 'JOIN' in query_upper:
            for rel in self.kg.relationships:
                if rel.left_table in tables and rel.right_table in tables:
                    joins.append((rel.left_table, rel.right_table))
        
        return {
            'tables': tables,
            'joins': joins,
            'where_conditions': where_conditions
        }
    
    def _reorder_joins(self, query: str, optimal_order: List[str]) -> str:
        """
        Reorder joins in the query based on optimal order.
        This is a simplified implementation.
        """
        # Add comment with suggested join order
        optimized = f"-- Suggested join order: {' -> '.join(optimal_order)}\n"
        optimized += query
        return optimized
    
    def _has_foreign_key_joins(self, query_components: Dict[str, List[str]]) -> bool:
        """Check if query uses foreign key joins."""
        for join_pair in query_components['joins']:
            table1, table2 = join_pair
            if self.kg.get_join_info(table1, table2):
                return True
        return False
    
    def _add_join_hints(self, query: str, query_components: Dict[str, List[str]]) -> str:
        """Add hints for efficient foreign key joins."""
        hints = []
        for join_pair in query_components['joins']:
            table1, table2 = join_pair
            join_info = self.kg.get_join_info(table1, table2)
            if join_info:
                hints.append(f"Use index on {join_info.left_table}.{join_info.left_column}")
        
        if hints:
            hint_comment = f"-- Join hints: {'; '.join(hints)}\n"
            return hint_comment + query
        
        return query
    

    def _add_index_hints(self, query: str, query_components: Dict[str, List[str]]) -> str:
        """Add index suggestions for WHERE conditions."""
        index_suggestions = []
        for condition in query_components['where_conditions']:
            index_suggestions.append(f"Consider index on {condition}")
        
        if index_suggestions:
            index_comment = f"-- Index suggestions: {'; '.join(index_suggestions)}\n"
            return index_comment + query
        
        return query
    

class BaselineEvaluator:
    """
    Evaluator for baseline optimizer performance.
    """
    
    def __init__(self, db_simulator: DatabaseSimulator, knowledge_graph: DatabaseSchemaKG):
        self.db = db_simulator
        self.kg = knowledge_graph
        self.rule_optimizer = RuleBasedOptimizer(knowledge_graph)

    def evaluate_baseline_performance(self, test_queries: List[str]) -> Dict[str, float]:
        """
        Evaluate baseline optimizer performance on test queries.
        
        Args:
            test_queries: List of SQL queries to test
            
        Returns:
            Performance metrics dictionary
        """
        logger.info(f"Evaluating baseline performance on {len(test_queries)} queries")
        
        original_times = []
        optimized_times = []
        optimization_results = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"Processing query {i+1}/{len(test_queries)}")
            
            # Execute original query
            original_result = self.db.execute_query(query)
            if original_result.error:
                logger.warning(f"Original query {i+1} failed: {original_result.error}")
                continue
            
            # Optimize query
            optimization = self.rule_optimizer.optimize_query(query)
            
            # Execute optimized query
            optimized_result = self.db.execute_query(optimization.optimized_query)
            if optimized_result.error:
                logger.warning(f"Optimized query {i+1} failed: {optimized_result.error}")
                continue
            
            # Store results
            original_times.append(original_result.execution_time)
            optimized_times.append(optimized_result.execution_time)
            
            optimization.execution_result = optimized_result
            optimization_results.append(optimization)
            
            # Log improvement
            improvement = ((original_result.execution_time - optimized_result.execution_time) / 
                          original_result.execution_time) * 100
            logger.info(f"Query {i+1}: {original_result.execution_time:.3f}s -> {optimized_result.execution_time:.3f}s ({improvement:+.1f}%)")

        # Calculate metrics
        if not original_times:
            return {"error": "No successful query executions"}
        
        avg_original_time = sum(original_times) / len(original_times)
        avg_optimized_time = sum(optimized_times) / len(optimized_times)
        avg_improvement = ((avg_original_time - avg_optimized_time) / avg_original_time) * 100
        
        successful_optimizations = len([opt for opt in optimization_results 
                                      if opt.estimated_improvement > 0])
        
        metrics = {
            "total_queries": len(test_queries),
            "successful_executions": len(original_times),
            "avg_original_time": avg_original_time,
            "avg_optimized_time": avg_optimized_time,
            "avg_improvement_percent": avg_improvement,
            "successful_optimizations": successful_optimizations,
            "optimization_success_rate": successful_optimizations / len(optimization_results) if optimization_results else 0
        }
        
        return metrics
    
    def print_evaluation_summary(self, metrics: Dict[str, float]):
        """Print a summary of the evaluation results."""
        print("\n=== Baseline Optimizer Evaluation Summary ===")
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        print(f"Total queries tested: {metrics['total_queries']}")
        print(f"Successful executions: {metrics['successful_executions']}")
        print(f"Average original execution time: {metrics['avg_original_time']:.3f}s")
        print(f"Average optimized execution time: {metrics['avg_optimized_time']:.3f}s")
        print(f"Average improvement: {metrics['avg_improvement_percent']:+.1f}%")
        print(f"Successful optimizations: {metrics['successful_optimizations']}")
        print(f"Optimization success rate: {metrics['optimization_success_rate']:.1%}")