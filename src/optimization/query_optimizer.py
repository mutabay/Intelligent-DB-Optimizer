"""
Core Query Optimizer Implementation

This module provides the main query optimization logic that integrates
multiple optimization strategies including rule-based, ML-based, and
reinforcement learning approaches.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..database_environment.db_simulator import DatabaseSimulator
from ..knowledge_graph.schema_ontology import DatabaseSchemaKG
from ..agents.dqn_agent import MultiAgentDQN
from ..agents.rl_environment import QueryOptimizationEnv
from ..utils.logging import logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    RULE_BASED = "rule_based"
    DQN_BASED = "dqn_based"
    HYBRID = "hybrid"


@dataclass
class QueryPlan:
    """Represents an optimized query execution plan."""
    query: str
    execution_plan: Dict[str, Any]
    estimated_cost: float
    optimization_strategy: OptimizationStrategy
    optimization_time: float
    metadata: Dict[str, Any]


class QueryOptimizer:
    """
    Main query optimizer that coordinates different optimization strategies.
    
    This class integrates rule-based optimization with reinforcement learning
    to provide adaptive query optimization capabilities.
    """
    
    def __init__(self, 
                 db_simulator: DatabaseSimulator,
                 knowledge_graph: DatabaseSchemaKG,
                 default_strategy: OptimizationStrategy = OptimizationStrategy.HYBRID):
        """
        Initialize the query optimizer.
        
        Args:
            db_simulator: Database simulator for execution
            knowledge_graph: Schema knowledge graph
            default_strategy: Default optimization strategy to use
        """
        self.db_simulator = db_simulator
        self.knowledge_graph = knowledge_graph
        self.default_strategy = default_strategy
        
        # Initialize RL components
        self.rl_env = None
        self.dqn_system = None
        self._initialize_rl_components()
        
        # Optimization history
        self.optimization_history = []
    
    def _initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        try:
            self.rl_env = QueryOptimizationEnv(
                db_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph
            )
            self.dqn_system = MultiAgentDQN()
            logger.info("RL components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RL components: {e}")
            self.rl_env = None
            self.dqn_system = None
    
    def optimize_query(self, 
                      query: str, 
                      strategy: Optional[OptimizationStrategy] = None) -> QueryPlan:
        """
        Optimize a SQL query using the specified strategy.
        
        Args:
            query: SQL query to optimize
            strategy: Optimization strategy to use (defaults to class default)
            
        Returns:
            Optimized query execution plan
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        logger.info(f"Optimizing query using {strategy.value} strategy")
        
        try:
            if strategy == OptimizationStrategy.RULE_BASED:
                plan = self._optimize_rule_based(query)
            elif strategy == OptimizationStrategy.DQN_BASED:
                plan = self._optimize_dqn_based(query)
            elif strategy == OptimizationStrategy.HYBRID:
                plan = self._optimize_hybrid(query)
            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
            
            optimization_time = time.time() - start_time
            plan.optimization_time = optimization_time
            
            # Store optimization history
            self.optimization_history.append({
                'timestamp': time.time(),
                'query': query,
                'strategy': strategy,
                'cost': plan.estimated_cost,
                'optimization_time': optimization_time
            })
            
            logger.info(f"Query optimized in {optimization_time:.3f}s using {strategy.value}")
            return plan
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            # Return fallback plan
            return self._create_fallback_plan(query, strategy, time.time() - start_time)
    
    def _optimize_rule_based(self, query: str) -> QueryPlan:
        """Optimize query using rule-based approach."""
        # Parse query to extract components
        query_analysis = self._analyze_query(query)
        
        # Apply rule-based optimizations
        optimization_decisions = {
            'join_order': self._suggest_join_order_rules(query_analysis),
            'index_usage': self._suggest_index_usage_rules(query_analysis),
            'scan_method': self._select_scan_method_rules(query_analysis)
        }
        
        # Estimate cost
        estimated_cost = self._estimate_plan_cost(optimization_decisions, query_analysis)
        
        return QueryPlan(
            query=query,
            execution_plan=optimization_decisions,
            estimated_cost=estimated_cost,
            optimization_strategy=OptimizationStrategy.RULE_BASED,
            optimization_time=0.0,  # Will be set by caller
            metadata={
                'query_analysis': query_analysis,
                'rule_applications': len(optimization_decisions)
            }
        )
    
    def _optimize_dqn_based(self, query: str) -> QueryPlan:
        """Optimize query using DQN-based approach."""
        if not self.rl_env or not self.dqn_system:
            logger.warning("RL components not available, falling back to rule-based")
            return self._optimize_rule_based(query)
        
        # Reset environment with query
        state = self.rl_env.reset(query=query)
        
        # Get DQN optimization decisions
        actions = self.dqn_system.get_actions(state, deterministic=True)
        
        # Apply actions to get execution plan
        next_state, reward, done, truncated, info = self.rl_env.step(actions)
        
        optimization_decisions = {
            'join_order_action': actions[0],
            'index_action': actions[1],
            'cache_action': actions[2],
            'resource_action': actions[3],
            'execution_plan': info.get('execution_plan', {}),
            'dqn_confidence': info.get('confidence', 0.0)
        }
        
        estimated_cost = info.get('cost_estimate', 1.0)
        
        return QueryPlan(
            query=query,
            execution_plan=optimization_decisions,
            estimated_cost=estimated_cost,
            optimization_strategy=OptimizationStrategy.DQN_BASED,
            optimization_time=0.0,  # Will be set by caller
            metadata={
                'dqn_actions': actions,
                'rl_reward': reward,
                'environment_info': info
            }
        )
    
    def _optimize_hybrid(self, query: str) -> QueryPlan:
        """Optimize query using hybrid rule-based + DQN approach."""
        # Get both optimization results
        rule_based_plan = self._optimize_rule_based(query)
        
        if self.rl_env and self.dqn_system:
            dqn_plan = self._optimize_dqn_based(query)
            
            # Select better plan based on estimated cost
            if dqn_plan.estimated_cost < rule_based_plan.estimated_cost:
                selected_plan = dqn_plan
                selected_plan.optimization_strategy = OptimizationStrategy.HYBRID
                selected_plan.metadata['hybrid_choice'] = 'dqn'
                selected_plan.metadata['rule_based_cost'] = rule_based_plan.estimated_cost
            else:
                selected_plan = rule_based_plan
                selected_plan.optimization_strategy = OptimizationStrategy.HYBRID
                selected_plan.metadata['hybrid_choice'] = 'rule_based'
                selected_plan.metadata['dqn_cost'] = dqn_plan.estimated_cost
        else:
            # Fall back to rule-based if RL not available
            selected_plan = rule_based_plan
            selected_plan.optimization_strategy = OptimizationStrategy.HYBRID
            selected_plan.metadata['hybrid_choice'] = 'rule_based_fallback'
        
        return selected_plan
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query structure and characteristics."""
        query_lower = query.lower().strip()
        
        # Basic query analysis
        analysis = {
            'query_type': 'SELECT',  # Could be enhanced with proper SQL parsing
            'has_joins': 'join' in query_lower,
            'has_subqueries': '(' in query and 'select' in query_lower.split('(')[1] if '(' in query else False,
            'has_aggregations': any(agg in query_lower for agg in ['sum', 'count', 'avg', 'max', 'min']),
            'has_group_by': 'group by' in query_lower,
            'has_order_by': 'order by' in query_lower,
            'estimated_selectivity': 0.5,  # Could be enhanced with statistics
            'table_count': query_lower.count('from') + query_lower.count('join')
        }
        
        # Extract table names (simplified)
        tables = []
        words = query_lower.split()
        for i, word in enumerate(words):
            if word == 'from' or word == 'join':
                if i + 1 < len(words):
                    table_name = words[i + 1].strip(',')
                    if table_name not in ['(', 'select']:
                        tables.append(table_name)
        
        analysis['tables'] = tables
        analysis['complexity_score'] = len(tables) + (2 if analysis['has_joins'] else 0) + \
                                     (1 if analysis['has_subqueries'] else 0) + \
                                     (1 if analysis['has_aggregations'] else 0)
        
        return analysis
    
    def _suggest_join_order_rules(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Suggest join order using rule-based heuristics."""
        tables = query_analysis.get('tables', [])
        
        if len(tables) <= 1:
            return tables
        
        # Simple heuristic: smaller tables first (would need actual statistics)
        # For now, use alphabetical order as placeholder
        return sorted(tables)
    
    def _suggest_index_usage_rules(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Suggest index usage using rule-based heuristics."""
        suggestions = []
        
        if query_analysis.get('has_joins'):
            suggestions.append('create_join_indexes')
        
        if query_analysis.get('has_aggregations'):
            suggestions.append('create_aggregation_indexes')
        
        if query_analysis.get('has_order_by'):
            suggestions.append('create_sort_indexes')
        
        return suggestions
    
    def _select_scan_method_rules(self, query_analysis: Dict[str, Any]) -> str:
        """Select scan method using rule-based heuristics."""
        if query_analysis.get('estimated_selectivity', 1.0) < 0.1:
            return 'index_scan'
        elif query_analysis.get('table_count', 0) > 3:
            return 'hash_join'
        else:
            return 'sequential_scan'
    
    def _estimate_plan_cost(self, optimization_decisions: Dict[str, Any], 
                           query_analysis: Dict[str, Any]) -> float:
        """Estimate execution cost for the optimization plan."""
        base_cost = query_analysis.get('complexity_score', 1) * 1000
        
        # Adjust cost based on optimization decisions
        if optimization_decisions.get('scan_method') == 'index_scan':
            base_cost *= 0.3  # Index scan is typically faster
        elif optimization_decisions.get('scan_method') == 'hash_join':
            base_cost *= 0.7  # Hash join is moderate
        
        # Add cost for index usage
        index_suggestions = optimization_decisions.get('index_usage', [])
        base_cost += len(index_suggestions) * 100  # Cost of index operations
        
        return max(base_cost, 100)  # Minimum cost threshold
    
    def _create_fallback_plan(self, query: str, strategy: OptimizationStrategy, 
                             optimization_time: float) -> QueryPlan:
        """Create a fallback plan when optimization fails."""
        return QueryPlan(
            query=query,
            execution_plan={'type': 'fallback', 'method': 'sequential_scan'},
            estimated_cost=10000.0,  # High cost to indicate suboptimal plan
            optimization_strategy=strategy,
            optimization_time=optimization_time,
            metadata={'error': 'optimization_failed', 'fallback': True}
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        if not self.optimization_history:
            return {'total_optimizations': 0}
        
        strategies = [h['strategy'] for h in self.optimization_history]
        costs = [h['cost'] for h in self.optimization_history]
        times = [h['optimization_time'] for h in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'strategy_distribution': {
                strategy.value: strategies.count(strategy) 
                for strategy in OptimizationStrategy
            },
            'average_cost': sum(costs) / len(costs),
            'average_optimization_time': sum(times) / len(times),
            'latest_optimizations': self.optimization_history[-5:]  # Last 5
        }