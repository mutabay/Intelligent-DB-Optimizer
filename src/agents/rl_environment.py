"""
Reinforcement Learning environment.
Multi-agent environment where different agents optimize different aspects.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.database_environment.db_simulator import DatabaseSimulator, QueryResult
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.utils.logging import logger

@dataclass
class QueryState:
    """Represents the current state of a query optimization problem"""
    tables: List[str]
    table_sizes: Dict[str, int]
    join_conditions: List[Tuple[str, str, str]]  # (table1, table2, condition)
    where_conditions: List[str]
    query_complexity: float  # e.g., estimated cost or complexity score
    historical_performance: Dict[str, float]  # e.g., past execution times

@dataclass
class OptimizationAction:
    """Represents an action taken by an optimization agent"""
    agent_type: str  # e.g., "indexing", "join_ordering", "query_rewriting"
    action_type: str  # e.g., "add_index", "change_join_order", "rewrite_query"
    parameters: Dict[str, Any]  # e.g., {"table": "orders", "column": "customer_id"}
    confidence: float  # Confidence score of the action

class QueryOptimizationEnv(gym.Env):
    """ Multi-agent RL environment for query optimization.
    Different agents can take actions to optimize different aspects of queries.
    """

    def __init__(self, database_simulator: DatabaseSimulator, knowledge_graph: DatabaseSchemaKG):
        super().__init__()

        self.db = database_simulator
        self.db_simulator = database_simulator  # Alias for compatibility with tests
        self.kg = knowledge_graph
        self.knowledge_graph = knowledge_graph  # Alias for compatibility with tests

        # Current query and state
        self.current_query = None
        self.current_state = None
        self.baseline_performance = None
        self.optimization_history = []

        # Multi-agent action space
        self.agent_types = ["join_ordering","index_advisor", "cache_manager", "resource_allocator"]

        # Define action spaces for each agent type
        self.action_spaces = {
            "join_ordering": spaces.Discrete(6),  # Different join order strategies
            "index_advisor": spaces.Discrete(4),  # Index strategies
            "cache_manager": spaces.Discrete(3),  # Cache strategies
            "resource_allocator": spaces.Discrete(3)  # Resource allocation strategies
        }

        # Combined action space (all agents can act)
        self.action_space = spaces.Dict(self.action_spaces)

        # Observation space: detailed query features + performance metrics
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # Performance tracking
        self.episode_rewards = []
        self.performance_history = []
        self.query_templates = self._initialize_query_templates()

        logger.info("Multi-agent RL environment initialized.")

    def debug_table_info(self):
        """Debug method to see what columns actually exist."""
        print("\n=== DEBUG: Table Information ===")
        for table_name, table_info in self.kg.tables.items():
            print(f"Table: {table_name}")
            print(f"  Columns: {table_info.columns}")
            print(f"  Primary Key: {table_info.primary_key}")
            print(f"  Foreign Keys: {table_info.foreign_keys}")
            print()

    def _initialize_query_templates(self) -> List[str]:
        """Initialize a set of query templates for training."""  
        return [
            # Single table queries
            "SELECT COUNT(*) FROM {table1} WHERE {where_condition}",
            "SELECT * FROM {table1} WHERE {where_condition} ORDER BY {order_column}",
            
            # Two table joins
            "SELECT {table1}.*, {table2}.* FROM {table1} JOIN {table2} ON {join_condition} WHERE {where_condition}",
            "SELECT COUNT(*) FROM {table1} JOIN {table2} ON {join_condition} WHERE {where_condition}",
            
            # Three table joins
            "SELECT {table1}.{col1}, {table2}.{col2}, {table3}.{col3} FROM {table1} JOIN {table2} ON {join_condition1} JOIN {table3} ON {join_condition2} WHERE {where_condition}",
            
            # Aggregation queries
            "SELECT {table1}.{group_col}, COUNT(*), AVG({table2}.{agg_col}) FROM {table1} JOIN {table2} ON {join_condition} GROUP BY {table1}.{group_col}",
        ]
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with a new query optimization challenge."""
        super().reset(seed=seed)
        
        # Generate or select a query for optimization
        self.current_query = self._generate_optimization_query()
        self.current_state = self._parse_query_to_state(self.current_query)
        
        # Get baseline performance (unoptimized query)
        baseline_result = self.db.execute_query(self.current_query)
        self.baseline_performance = baseline_result.execution_time if not baseline_result.error else 10.0
        # Ensure baseline is never zero to avoid division by zero
        if self.baseline_performance <= 0:
            self.baseline_performance = 0.001
        
        # Reset optimization history
        self.optimization_history = []
        
        observation = self._get_observation()
        info = {
            "query": self.current_query,
            "baseline_performance": self.baseline_performance,
            "state": self.current_state.__dict__,
            "tables_involved": self.current_state.tables,
            "join_count": len(self.current_state.join_conditions),
            "complexity": self.current_state.query_complexity
        }
        
        logger.info(f"Environment reset with query complexity: {self.current_state.query_complexity}")
        
        return observation, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute optimization actions from multiple agents.
        
        Args:
            actions: Dict of actions from each agent type
            
        Returns:
            observation, reward, done, truncated, info
        """
        
        # Apply optimizations from each agent
        optimizations = []
        for agent_type, action in actions.items():
            if agent_type in self.agent_types:
                optimization = self._apply_agent_action(agent_type, action)
                if optimization:
                    optimizations.append(optimization)

        # Generate optimized query
        optimized_query = self._apply_optimizations(self.current_query, optimizations)

        # Execute optimized query and measure performance
        result = self.db.execute_query(optimized_query)

        if result.error:
            # Penalize for creating invalid queries
            reward = -10.0
            performance = self.baseline_performance * 2  # Arbitrary high penalty
            improvement = -1.0
            logger.warning(f"Optimized query failed: {result.error}")
        
        else:
            performance = result.execution_time
            # Calculate improvement over baseline (positive for better performance)
            # Ensure we don't divide by zero
            if self.baseline_performance > 0:
                improvement = (self.baseline_performance - performance) / self.baseline_performance
            else:
                improvement = 0.0

            # Reward function: improvement *10 + bonus for significant improvements
            reward = improvement * 10.0
            if improvement > 0.2:
                reward += 5.0  # Bonus for >20% improvement
            elif improvement < -0.1:    # Penalty for > 10% degradation
                reward -= 3.0

        # Update performance history
        self.performance_history.append(performance)
        self.optimization_history.extend(optimizations)
        self.episode_rewards.append(reward)

        # Episode is done after applying 
        done = True

        observation = self._get_observation()
        info = {
            "optimizations_applied": [opt.__dict__ for opt in optimizations],
            "optimized_query": optimized_query,
            "performance": performance,
            "improvement": improvement,
            "baseline_performance": self.baseline_performance,
            "reward": reward,
            "query_valid": not bool(result.error),
            "optimization_count": len(optimizations),
            "cost_estimate": performance,  # Add cost estimate for compatibility
            "execution_time": performance,  # Add execution time for compatibility
            "actions_taken": list(actions.keys())  # Add actions taken for compatibility
        }

        logger.info(f"Step completed: improvement={improvement:.3f}, reward={reward:.2f}")
        
        return observation, reward, done, False, info
    
    def _generate_optimization_query(self) -> str:
        """Generate a query that needs optimization."""
        tables = list(self.kg.tables.keys())
        
        # Debug: Let's see what we're working with
        logger.info(f"Available tables: {tables}")
        for table in tables:
            table_info = self.kg.get_table_info(table)
            logger.info(f"Table {table} columns: {table_info.columns}")
        
        if len(tables) >= 3:
            # Complex 3-table join query
            table1, table2, table3 = tables[0], tables[1], tables[2]  # customers, orders, lineitem
            
            # Get actual column names from knowledge graph
            t1_info = self.kg.get_table_info(table1)  # customers
            t2_info = self.kg.get_table_info(table2)  # orders
            t3_info = self.kg.get_table_info(table3)  # lineitem
            
            # Build realistic join conditions from relationships
            join_cond1 = self._get_join_condition(table1, table2)  # customers -> orders
            join_cond2 = self._get_join_condition(table2, table3)  # orders -> lineitem
            
            if join_cond1 and join_cond2:
                # FIXED: Select appropriate columns from EACH table
                # Make sure we select columns that actually exist in each table
                
                # From customers table: select name (usually second column)
                customer_col = t1_info.columns[1] if len(t1_info.columns) > 1 else t1_info.columns[0]
                
                # From orders table: select a meaningful column (like totalprice)
                order_col = None
                for col in t2_info.columns:
                    if 'price' in col.lower() or 'total' in col.lower():
                        order_col = col
                        break
                if not order_col:
                    order_col = t2_info.columns[1] if len(t2_info.columns) > 1 else t2_info.columns[0]
                
                # From lineitem table: select quantity or similar
                lineitem_col = None
                for col in t3_info.columns:
                    if 'quantity' in col.lower() or 'qty' in col.lower():
                        lineitem_col = col
                        break
                if not lineitem_col:
                    lineitem_col = t3_info.columns[1] if len(t3_info.columns) > 1 else t3_info.columns[0]
                
                query = f"""SELECT {table1[:1]}.{customer_col}, {table2[:1]}.{order_col}, {table3[:1]}.{lineitem_col}
                    FROM {table1} {table1[:1]}
                    JOIN {table2} {table2[:1]} ON {join_cond1}
                    JOIN {table3} {table3[:1]} ON {join_cond2}
                    WHERE {table1[:1]}.{self._get_filterable_column(table1)} = 'BUILDING'
                    AND {table2[:1]}.{self._get_numeric_column(table2)} > 1000
                    ORDER BY {table2[:1]}.{self._get_numeric_column(table2)} DESC""".strip()
                
                logger.info(f"Generated 3-table query: {query}")
                return query
                            
        elif len(tables) >= 2:
            # Simple 2-table join
            table1, table2 = tables[0], tables[1]
            join_cond = self._get_join_condition(table1, table2)
            
            if join_cond:
                query = f"""SELECT {table1[:1]}.*, {table2[:1]}.*
                    FROM {table1} {table1[:1]}
                    JOIN {table2} {table2[:1]} ON {join_cond}
                    WHERE {table1[:1]}.{self._get_filterable_column(table1)} = 'BUILDING'""".strip()
                
                logger.info(f"Generated 2-table query: {query}")
                return query
        
        # Fallback: single table query
        if tables:
            table = tables[0]
            query = f"""SELECT COUNT(*) FROM {table} WHERE {self._get_filterable_column(table)} = 'BUILDING'""".strip()
            logger.info(f"Generated single-table query: {query}")
            return query
        
        # Ultimate fallback
        return "SELECT 1"
    
    def _get_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """Get join condition between two tables from knowledge graph."""
        join_info = self.kg.get_join_info(table1, table2)
        if join_info:
            # Handle the join direction correctly
            if join_info.left_table == table1:
                # table1 -> table2: table1.left_col = table2.right_col
                return f"{table1[:1]}.{join_info.left_column} = {table2[:1]}.{join_info.right_column}"
            else:
                # table2 -> table1: table1.right_col = table2.left_col
                return f"{table1[:1]}.{join_info.right_column} = {table2[:1]}.{join_info.left_column}"
        return None
    
    def _get_filterable_column(self, table: str) -> str:
        """Get a column suitable for WHERE conditions."""
        table_info = self.kg.get_table_info(table)
        if table_info and table_info.columns:
            # Look for common filterable columns
            for col in table_info.columns:
                if 'segment' in col.lower() or 'status' in col.lower() or 'type' in col.lower():
                    return col
            # Fallback to second column if available, otherwise first
            if len(table_info.columns) > 1:
                return table_info.columns[1]
            else:
                return table_info.columns[0]
        return "id"  # Ultimate fallback

    def _get_numeric_column(self, table: str) -> str:
        """Get a numeric column for comparisons."""
        table_info = self.kg.get_table_info(table)
        if table_info and table_info.columns:
            # Look for common numeric columns
            for col in table_info.columns:
                if 'price' in col.lower() or 'amount' in col.lower() or 'total' in col.lower():
                    return col
            # Look for columns with 'bal' (balance) or 'key' (might be numeric)
            for col in table_info.columns:
                if 'bal' in col.lower():
                    return col
            # Fallback to primary key or first column
            if table_info.primary_key:
                return table_info.primary_key[0]
            else:
                return table_info.columns[0]
        return "id"
    
    def _parse_query_to_state(self, query: str) -> QueryState:
        """Parse query into comprehensive state representation."""
        query_upper = query.upper()
        
        # Extract tables involved
        tables = []
        for table in self.kg.tables.keys():
            if table.upper() in query_upper or f" {table[:1].upper()}." in query_upper:
                tables.append(table)
        
        # Get table sizes
        table_sizes = {}
        for table in tables:  # Now 'tables' is defined!
            table_info = self.kg.get_table_info(table)
            table_sizes[table] = table_info.row_count if table_info else 100
        
        
        # Extract join conditions with full details
        join_conditions = []
        for rel in self.kg.relationships:
            if rel.left_table in tables and rel.right_table in tables:
                condition = f"{rel.left_column} = {rel.right_column}"
                join_conditions.append((rel.left_table, rel.right_table, condition))
        
        # Extract WHERE conditions
        where_conditions = []
        if "WHERE" in query_upper:
            # Parse common WHERE patterns
            lines = query_upper.split("WHERE")[1].split("ORDER")[0] if "ORDER" in query_upper else query_upper.split("WHERE")[1]
            
            if "MKTSEGMENT" in lines:
                where_conditions.append("mktsegment = 'BUILDING'")
            if "TOTALPRICE" in lines and ">" in lines:
                where_conditions.append("totalprice > 1000")
            if "=" in lines and not any(x in lines for x in ["MKTSEGMENT", "TOTALPRICE"]):
                where_conditions.append("generic_equality_condition")
        
        # Calculate complexity score
        complexity = (
            len(tables) * 2.0 +
            len(join_conditions) * 3.0 +
            len(where_conditions) * 1.5 +
            (1.0 if "ORDER BY" in query_upper else 0.0) +
            (1.0 if "GROUP BY" in query_upper else 0.0) +
            (2.0 if "COUNT" in query_upper else 0.0)
        )
        
        # Get historical performance data
        historical_performance = {}
        if len(self.performance_history) > 0:
            recent_performance = self.performance_history[-10:]
            historical_performance["avg_time"] = np.mean(recent_performance)
            historical_performance["best_time"] = min(self.performance_history)
            historical_performance["worst_time"] = max(self.performance_history)
            historical_performance["std_time"] = np.std(recent_performance)
        else:
            historical_performance = {"avg_time": 1.0, "best_time": 1.0, "worst_time": 1.0, "std_time": 0.0}
        
        return QueryState(
            tables=tables,
            table_sizes=table_sizes,
            join_conditions=join_conditions,
            where_conditions=where_conditions,
            query_complexity=complexity,
            historical_performance=historical_performance
        )
    
    def _apply_agent_action(self, agent_type: str, action: int) -> Optional[OptimizationAction]:
        """Apply action from a specific agent type."""
        
        if agent_type == "join_ordering":
            return self._apply_join_ordering_action(action)
        elif agent_type == "index_advisor":
            return self._apply_index_advisor_action(action)
        elif agent_type == "cache_manager":
            return self._apply_cache_manager_action(action)
        elif agent_type == "resource_allocator":
            return self._apply_resource_allocator_action(action)
        
        return None
    
    def _apply_join_ordering_action(self, action: int) -> OptimizationAction:
        """Apply join ordering optimization based on action."""
        
        tables = self.current_state.tables.copy()
        
        if action == 0:  # Smallest table first
            ordered_tables = sorted(tables, key=lambda t: self.current_state.table_sizes.get(t, 0))
            strategy = "smallest_first"
        elif action == 1:  # Largest table first
            ordered_tables = sorted(tables, key=lambda t: self.current_state.table_sizes.get(t, 0), reverse=True)
            strategy = "largest_first"
        elif action == 2:  # Knowledge graph suggestion
            ordered_tables = self.kg.suggest_join_order(tables)
            strategy = "knowledge_graph"
        elif action == 3:  # Foreign key driven order
            ordered_tables = self._get_fk_driven_order(tables)
            strategy = "foreign_key_driven"
        elif action == 4:  # Random order (exploration)
            ordered_tables = tables.copy()
            np.random.shuffle(ordered_tables)
            strategy = "random"
        else:  # action == 5, keep original order
            ordered_tables = tables
            strategy = "original"
        
        confidence = 0.8 if action in [0, 2, 3] else 0.5  # Higher confidence for proven strategies
        
        return OptimizationAction(
            agent_type="join_ordering",
            action_type="reorder_joins",
            parameters={"join_order": ordered_tables, "strategy": strategy},
            confidence=confidence
        )
    
    def _apply_index_advisor_action(self, action: int) -> OptimizationAction:
        """Apply index advisor optimization."""
        
        if action == 0:  # Index on WHERE columns
            strategy = "where_columns"
            target_columns = [cond.split()[0] for cond in self.current_state.where_conditions]
        elif action == 1:  # Index on JOIN columns
            strategy = "join_columns"
            target_columns = []
            for _, _, condition in self.current_state.join_conditions:
                if "=" in condition:
                    cols = condition.split("=")
                    target_columns.extend([col.strip() for col in cols])
        elif action == 2:  # Composite indexes (WHERE + JOIN)
            strategy = "composite"
            target_columns = []
            # Combine WHERE and JOIN columns
            for cond in self.current_state.where_conditions:
                target_columns.append(cond.split()[0])
            for _, _, condition in self.current_state.join_conditions:
                if "=" in condition:
                    cols = condition.split("=")
                    target_columns.extend([col.strip() for col in cols])
        else:  # action == 3, no additional indexes
            strategy = "none"
            target_columns = []
        
        return OptimizationAction(
            agent_type="index_advisor",
            action_type="suggest_indexes",
            parameters={"strategy": strategy, "target_columns": target_columns[:5]},  # Limit to 5 columns
            confidence=0.7
        )
    
    def _apply_cache_manager_action(self, action: int) -> OptimizationAction:
        """Apply cache management optimization."""
        
        if action == 0:  # Enable aggressive caching
            strategy = "aggressive_cache"
            cache_ttl = 300  # 5 minutes
        elif action == 1:  # Disable caching
            strategy = "no_cache"
            cache_ttl = 0
        else:  # action == 2, adaptive caching based on query complexity
            if self.current_state.query_complexity > 8:
                strategy = "adaptive_high"
                cache_ttl = 600  # 10 minutes for complex queries
            else:
                strategy = "adaptive_low"
                cache_ttl = 60  # 1 minute for simple queries
        
        return OptimizationAction(
            agent_type="cache_manager",
            action_type="manage_cache",
            parameters={"strategy": strategy, "ttl": cache_ttl},
            confidence=0.6
        )
    
    def _apply_resource_allocator_action(self, action: int) -> OptimizationAction:
        """Apply resource allocation optimization."""
        
        if action == 0:  # High memory allocation for large joins
            strategy = "high_memory"
            memory_factor = 2.0
            cpu_factor = 1.0
        elif action == 1:  # Balanced allocation
            strategy = "balanced"
            memory_factor = 1.0
            cpu_factor = 1.0
        else:  # action == 2, low memory/high CPU for small data
            strategy = "cpu_intensive"
            memory_factor = 0.5
            cpu_factor = 2.0
        
        return OptimizationAction(
            agent_type="resource_allocator",
            action_type="allocate_resources",
            parameters={
                "strategy": strategy, 
                "memory_factor": memory_factor, 
                "cpu_factor": cpu_factor
            },
            confidence=0.5
        )
    
    def _get_fk_driven_order(self, tables: List[str]) -> List[str]:
        """Get join order based on foreign key relationships."""
        if len(tables) <= 1:
            return tables
        
        # Build dependency graph
        dependencies = {}
        for table in tables:
            dependencies[table] = set()
        
        # Add dependencies based on foreign keys
        for rel in self.kg.relationships:
            if rel.left_table in tables and rel.right_table in tables:
                dependencies[rel.left_table].add(rel.right_table)
        
        # Topological sort
        ordered = []
        remaining = set(tables)
        
        while remaining:
            # Find table with no unresolved dependencies
            for table in remaining:
                if not (dependencies[table] & remaining):
                    ordered.append(table)
                    remaining.remove(table)
                    break
            else:
                # If no table without dependencies, add any remaining (circular dependency)
                ordered.append(remaining.pop())
        
        return ordered
    
    def _apply_optimizations(self, original_query: str, optimizations: List[OptimizationAction]) -> str:
        if not optimizations:
            return original_query
        
        hints = [
            "-- Multi-Agent RL Optimizations Applied --"
        ]
        
        for opt in optimizations:
            hints.append(f"-- Agent: {opt.agent_type} (confidence: {opt.confidence:.2f})")
            
            if opt.action_type == "reorder_joins":
                join_order = opt.parameters.get('join_order', [])
                strategy = opt.parameters.get('strategy', 'unknown')
                hints.append(f"--   Join Strategy: {strategy}")
                hints.append(f"--   Optimal Join Order: {' -> '.join(join_order)}")
                
                # Add specific join condition details
                for table1, table2, condition in self.current_state.join_conditions:
                    hints.append(f"--   Join Condition: {table1} JOIN {table2} ON {condition}")
            
            elif opt.action_type == "suggest_indexes":
                strategy = opt.parameters.get('strategy', 'none')
                target_columns = opt.parameters.get('target_columns', [])
                
                if strategy != "none":
                    hints.append(f"--   Index Strategy: {strategy}")
                    if target_columns:
                        hints.append(f"--   Target Columns: {', '.join(target_columns[:3])}")
                        
                        # Generate specific CREATE INDEX statements
                        for i, col in enumerate(target_columns[:3]):
                            table_name = self._extract_table_from_column(col)
                            if table_name:
                                col_name = col.split('.')[-1] if '.' in col else col
                                hints.append(f"--   CREATE INDEX idx_{table_name}_{col_name} ON {table_name}({col_name});")
            
            elif opt.action_type == "manage_cache":
                strategy = opt.parameters.get('strategy', 'none')
                ttl = opt.parameters.get('ttl', 0)
                hints.append(f"--   Cache Strategy: {strategy}")
                if ttl > 0:
                    hints.append(f"--   Cache TTL: {ttl} seconds")
            
            elif opt.action_type == "allocate_resources":
                strategy = opt.parameters.get('strategy', 'balanced')
                mem_factor = opt.parameters.get('memory_factor', 1.0)
                cpu_factor = opt.parameters.get('cpu_factor', 1.0)
                hints.append(f"--   Resource Strategy: {strategy}")
                hints.append(f"--   Memory Factor: {mem_factor}x, CPU Factor: {cpu_factor}x")
        
        hints.append("-- End Optimizations --")
        hints.append("")
        
        return "\n".join(hints) + original_query
    
    def _extract_table_from_column(self, column: str) -> Optional[str]:
        """Extract table name from column reference."""
        if '.' in column:
            alias_or_table = column.split('.')[0]
            # Try to match with actual table names
            for table in self.current_state.tables:
                if table.startswith(alias_or_table.lower()) or alias_or_table.lower() == table[:1]:
                    return table
        
        # Fallback: find table that contains this column
        col_name = column.split('.')[-1]
        for table in self.current_state.tables:
            table_info = self.kg.get_table_info(table)
            if table_info and col_name in table_info.columns:
                return table
        
        return None
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector for RL agents."""
        
        if not self.current_state:
            return np.zeros(12, dtype=np.float32)
        
        # Comprehensive feature vector
        table_sizes = list(self.current_state.table_sizes.values())
        
        features = [
            len(self.current_state.tables),  # Number of tables
            len(self.current_state.join_conditions),  # Number of joins
            len(self.current_state.where_conditions),  # Number of WHERE conditions
            self.current_state.query_complexity,  # Query complexity score
            np.mean(table_sizes) if table_sizes else 0,  # Average table size
            max(table_sizes) if table_sizes else 0,  # Largest table size
            min(table_sizes) if table_sizes else 0,  # Smallest table size
            self.current_state.historical_performance.get("avg_time", 1.0),  # Historical avg performance
            self.current_state.historical_performance.get("best_time", 1.0),  # Historical best performance
            self.baseline_performance or 1.0,  # Current baseline performance
            len(self.performance_history) / 100.0,  # Experience factor (scaled)
            len(self.optimization_history) / 10.0  # Optimization history factor (scaled)
        ]
        
        # Normalize features to [0, 1] range
        normalized_features = []
        for i, feature in enumerate(features):
            if i < 4:  # Counts and complexity
                normalized_features.append(min(feature / 15.0, 1.0))
            elif i < 10:  # Performance and size metrics
                normalized_features.append(min(feature / 10.0, 1.0))
            else:  # Experience factors
                normalized_features.append(min(feature, 1.0))
        
        return np.array(normalized_features, dtype=np.float32)
    

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics for evaluation."""
        if not self.performance_history:
            return {
                "episodes": 0,
                "avg_performance": 0.0,
                "best_performance": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0
            }
        
        improvements = []
        if self.baseline_performance:
            for perf in self.performance_history:
                improvement = (self.baseline_performance - perf) / self.baseline_performance
                improvements.append(improvement)
        
        successful_episodes = len([r for r in self.episode_rewards if r > 0])
        
        return {
            "episodes": len(self.performance_history),
            "avg_performance": np.mean(self.performance_history),
            "best_performance": min(self.performance_history),
            "worst_performance": max(self.performance_history),
            "performance_std": np.std(self.performance_history),
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_improvement": np.mean(improvements) if improvements else 0.0,
            "best_improvement": max(improvements) if improvements else 0.0,
            "success_rate": successful_episodes / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "query_complexity_avg": np.mean([state.query_complexity for state in [self.current_state] if state]),
            "total_optimizations": len(self.optimization_history)
        }
    
    def render(self, mode='human'):
        """Render environment state (for debugging)."""
        if mode == 'human':
            print(f"\n=== Query Optimization Environment State ===")
            if self.current_state:
                print(f"Tables: {self.current_state.tables}")
                print(f"Join Conditions: {len(self.current_state.join_conditions)}")
                print(f"Where Conditions: {len(self.current_state.where_conditions)}")
                print(f"Complexity: {self.current_state.query_complexity:.2f}")
                print(f"Baseline Performance: {self.baseline_performance:.3f}s")
            
            if self.performance_history:
                print(f"Performance History: {len(self.performance_history)} episodes")
                print(f"Average Performance: {np.mean(self.performance_history):.3f}s")
                print(f"Best Performance: {min(self.performance_history):.3f}s")
            
            print(f"Optimizations Applied: {len(self.optimization_history)}")
            print("=" * 50)
    
    def extract_state_from_query(self, query: str) -> np.ndarray:
        """Extract state representation from query text.
        
        Args:
            query: SQL query text
            
        Returns:
            State vector representation
        """
        # Parse query to extract features
        query_lower = query.lower()
        
        # Count query features
        num_tables = len([word for word in query_lower.split() if word in ['from', 'join']])
        num_joins = query_lower.count('join')
        num_where = query_lower.count('where')
        num_select = query_lower.count('select')
        
        # Estimate complexity based on query structure
        complexity = num_tables * 2 + num_joins * 3 + num_where * 1.5 + num_select
        
        # Create state vector
        state = np.array([
            num_tables,
            num_joins, 
            num_where,
            num_select,
            complexity,
            len(query),  # Query length
            query_lower.count('group by'),
            query_lower.count('order by'),
            query_lower.count('having'),
            query_lower.count('limit')
        ], dtype=np.float32)
        
        # Normalize values to [0, 1] range
        max_values = np.array([10, 10, 10, 5, 50, 1000, 5, 5, 5, 5])  # Expected max values
        state = np.minimum(state / max_values, 1.0)  # Clip at 1.0
        
        # Pad or truncate to match observation space
        if len(state) > self.observation_space.shape[0]:
            state = state[:self.observation_space.shape[0]]
        elif len(state) < self.observation_space.shape[0]:
            state = np.pad(state, (0, self.observation_space.shape[0] - len(state)))
        
        return state