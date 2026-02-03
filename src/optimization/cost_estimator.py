"""
Cost Estimation Module

This module provides sophisticated cost estimation capabilities for
database query optimization, incorporating statistical models and
machine learning techniques.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import logger


class OperationType(Enum):
    """Types of database operations for cost estimation."""
    SEQUENTIAL_SCAN = "seq_scan"
    INDEX_SCAN = "index_scan"
    NESTED_LOOP_JOIN = "nested_loop"
    HASH_JOIN = "hash_join"
    SORT_MERGE_JOIN = "sort_merge"
    AGGREGATION = "aggregation"
    SORTING = "sorting"


@dataclass
class OperationCost:
    """Cost breakdown for a database operation."""
    cpu_cost: float
    io_cost: float
    memory_cost: float
    network_cost: float
    total_cost: float
    cardinality_estimate: int
    operation_type: OperationType


class CostModel:
    """Statistical cost model for database operations."""
    
    def __init__(self):
        """Initialize cost model with default parameters."""
        # Base cost parameters (can be tuned based on hardware)
        self.cpu_cost_per_tuple = 0.01
        self.io_cost_per_page = 1.0
        self.memory_cost_per_kb = 0.001
        self.network_cost_per_kb = 0.1
        
        # Operation-specific multipliers
        self.operation_multipliers = {
            OperationType.SEQUENTIAL_SCAN: 1.0,
            OperationType.INDEX_SCAN: 0.3,
            OperationType.NESTED_LOOP_JOIN: 2.0,
            OperationType.HASH_JOIN: 1.5,
            OperationType.SORT_MERGE_JOIN: 1.8,
            OperationType.AGGREGATION: 1.2,
            OperationType.SORTING: 1.4
        }
    
    def estimate_scan_cost(self, table_size: int, selectivity: float, 
                          use_index: bool = False) -> OperationCost:
        """Estimate cost for table scan operations."""
        operation_type = OperationType.INDEX_SCAN if use_index else OperationType.SEQUENTIAL_SCAN
        
        # Estimate cardinality
        cardinality = int(table_size * selectivity)
        
        if use_index:
            # Index scan: logarithmic I/O cost + sequential read of selected rows
            io_cost = math.log2(max(table_size, 1)) + cardinality * 0.1
            cpu_cost = cardinality * self.cpu_cost_per_tuple * 0.5  # Less CPU for index
        else:
            # Sequential scan: linear I/O and CPU
            io_cost = table_size * 0.01  # Assume 1% of rows per page
            cpu_cost = table_size * self.cpu_cost_per_tuple
        
        # Apply operation multiplier
        multiplier = self.operation_multipliers[operation_type]
        io_cost *= multiplier
        cpu_cost *= multiplier
        
        # Memory cost (buffer pool usage)
        memory_cost = min(table_size * 0.001, 1000) * self.memory_cost_per_kb
        
        total_cost = io_cost * self.io_cost_per_page + cpu_cost + memory_cost
        
        return OperationCost(
            cpu_cost=cpu_cost,
            io_cost=io_cost * self.io_cost_per_page,
            memory_cost=memory_cost,
            network_cost=0.0,
            total_cost=total_cost,
            cardinality_estimate=cardinality,
            operation_type=operation_type
        )
    
    def estimate_join_cost(self, left_cardinality: int, right_cardinality: int,
                          join_type: OperationType, selectivity: float = 0.1) -> OperationCost:
        """Estimate cost for join operations."""
        if join_type not in [OperationType.NESTED_LOOP_JOIN, OperationType.HASH_JOIN, 
                           OperationType.SORT_MERGE_JOIN]:
            raise ValueError(f"Invalid join type: {join_type}")
        
        # Estimate output cardinality
        output_cardinality = int(left_cardinality * right_cardinality * selectivity)
        
        if join_type == OperationType.NESTED_LOOP_JOIN:
            # Nested loop: O(M*N) complexity
            cpu_cost = left_cardinality * right_cardinality * self.cpu_cost_per_tuple
            io_cost = left_cardinality * (right_cardinality * 0.01)  # Inner table scans
            memory_cost = max(left_cardinality, right_cardinality) * 0.001
            
        elif join_type == OperationType.HASH_JOIN:
            # Hash join: O(M+N) complexity
            cpu_cost = (left_cardinality + right_cardinality) * self.cpu_cost_per_tuple * 2
            io_cost = (left_cardinality + right_cardinality) * 0.01
            # Hash table memory requirement
            memory_cost = min(left_cardinality, right_cardinality) * 0.01  
            
        else:  # SORT_MERGE_JOIN
            # Sort-merge: O(M*log(M) + N*log(N)) complexity
            sort_cost_left = left_cardinality * math.log2(max(left_cardinality, 1))
            sort_cost_right = right_cardinality * math.log2(max(right_cardinality, 1))
            cpu_cost = (sort_cost_left + sort_cost_right + left_cardinality + right_cardinality) * self.cpu_cost_per_tuple
            io_cost = (left_cardinality + right_cardinality) * 0.02  # Sort requires more I/O
            memory_cost = (left_cardinality + right_cardinality) * 0.005  # Sort buffers
        
        # Apply operation multiplier
        multiplier = self.operation_multipliers[join_type]
        cpu_cost *= multiplier
        io_cost *= multiplier
        
        total_cost = cpu_cost + io_cost * self.io_cost_per_page + memory_cost
        
        return OperationCost(
            cpu_cost=cpu_cost,
            io_cost=io_cost * self.io_cost_per_page,
            memory_cost=memory_cost,
            network_cost=0.0,
            total_cost=total_cost,
            cardinality_estimate=output_cardinality,
            operation_type=join_type
        )
    
    def estimate_aggregation_cost(self, input_cardinality: int, 
                                 group_count: int = None) -> OperationCost:
        """Estimate cost for aggregation operations."""
        if group_count is None:
            group_count = int(math.sqrt(input_cardinality))  # Heuristic
        
        # Aggregation typically requires sorting or hashing
        cpu_cost = input_cardinality * math.log2(max(group_count, 1)) * self.cpu_cost_per_tuple
        io_cost = input_cardinality * 0.01
        memory_cost = group_count * 0.001  # Memory for group state
        
        # Apply operation multiplier
        multiplier = self.operation_multipliers[OperationType.AGGREGATION]
        cpu_cost *= multiplier
        
        total_cost = cpu_cost + io_cost * self.io_cost_per_page + memory_cost
        
        return OperationCost(
            cpu_cost=cpu_cost,
            io_cost=io_cost * self.io_cost_per_page,
            memory_cost=memory_cost,
            network_cost=0.0,
            total_cost=total_cost,
            cardinality_estimate=group_count,
            operation_type=OperationType.AGGREGATION
        )


class CostEstimator:
    """
    Advanced cost estimator for database query optimization.
    
    This class provides comprehensive cost estimation capabilities
    incorporating statistical models and learned cost functions.
    """
    
    def __init__(self):
        """Initialize the cost estimator."""
        self.cost_model = CostModel()
        self.table_statistics = {}
        self.estimation_history = []
    
    def register_table_statistics(self, table_name: str, statistics: Dict[str, Any]):
        """
        Register table statistics for cost estimation.
        
        Args:
            table_name: Name of the table
            statistics: Dictionary containing table statistics
        """
        self.table_statistics[table_name] = {
            'row_count': statistics.get('row_count', 1000),
            'page_count': statistics.get('page_count', 100),
            'average_row_size': statistics.get('average_row_size', 100),
            'column_statistics': statistics.get('column_statistics', {}),
            'index_statistics': statistics.get('index_statistics', {})
        }
        logger.info(f"Registered statistics for table {table_name}")
    
    def estimate_query_cost(self, execution_plan: Dict[str, Any]) -> Tuple[float, List[OperationCost]]:
        """
        Estimate total cost for a query execution plan.
        
        Args:
            execution_plan: Dictionary describing the execution plan
            
        Returns:
            Tuple of (total_cost, operation_costs_list)
        """
        operation_costs = []
        
        # Process different types of operations in the plan
        if 'scan_operations' in execution_plan:
            for scan_op in execution_plan['scan_operations']:
                cost = self._estimate_scan_operation_cost(scan_op)
                operation_costs.append(cost)
        
        if 'join_operations' in execution_plan:
            for join_op in execution_plan['join_operations']:
                cost = self._estimate_join_operation_cost(join_op)
                operation_costs.append(cost)
        
        if 'aggregation_operations' in execution_plan:
            for agg_op in execution_plan['aggregation_operations']:
                cost = self._estimate_aggregation_operation_cost(agg_op)
                operation_costs.append(cost)
        
        # Calculate total cost
        total_cost = sum(op_cost.total_cost for op_cost in operation_costs)
        
        # Store estimation for learning
        self.estimation_history.append({
            'plan': execution_plan,
            'estimated_cost': total_cost,
            'operation_costs': operation_costs
        })
        
        return total_cost, operation_costs
    
    def _estimate_scan_operation_cost(self, scan_operation: Dict[str, Any]) -> OperationCost:
        """Estimate cost for a scan operation."""
        table_name = scan_operation.get('table', 'unknown')
        selectivity = scan_operation.get('selectivity', 1.0)
        use_index = scan_operation.get('use_index', False)
        
        # Get table statistics
        table_stats = self.table_statistics.get(table_name, {
            'row_count': 1000,
            'page_count': 100
        })
        
        table_size = table_stats['row_count']
        
        return self.cost_model.estimate_scan_cost(table_size, selectivity, use_index)
    
    def _estimate_join_operation_cost(self, join_operation: Dict[str, Any]) -> OperationCost:
        """Estimate cost for a join operation."""
        left_table = join_operation.get('left_table', 'unknown')
        right_table = join_operation.get('right_table', 'unknown')
        join_type_str = join_operation.get('join_type', 'hash_join')
        selectivity = join_operation.get('selectivity', 0.1)
        
        # Map string to enum
        join_type_map = {
            'nested_loop': OperationType.NESTED_LOOP_JOIN,
            'hash_join': OperationType.HASH_JOIN,
            'sort_merge': OperationType.SORT_MERGE_JOIN
        }
        join_type = join_type_map.get(join_type_str, OperationType.HASH_JOIN)
        
        # Get cardinalities
        left_stats = self.table_statistics.get(left_table, {'row_count': 1000})
        right_stats = self.table_statistics.get(right_table, {'row_count': 1000})
        
        left_cardinality = left_stats['row_count']
        right_cardinality = right_stats['row_count']
        
        return self.cost_model.estimate_join_cost(
            left_cardinality, right_cardinality, join_type, selectivity
        )
    
    def _estimate_aggregation_operation_cost(self, agg_operation: Dict[str, Any]) -> OperationCost:
        """Estimate cost for an aggregation operation."""
        input_cardinality = agg_operation.get('input_cardinality', 1000)
        group_count = agg_operation.get('group_count', None)
        
        return self.cost_model.estimate_aggregation_cost(input_cardinality, group_count)
    
    def compare_plans(self, plan_a: Dict[str, Any], plan_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare costs of two execution plans.
        
        Args:
            plan_a: First execution plan
            plan_b: Second execution plan
            
        Returns:
            Dictionary containing comparison results
        """
        cost_a, ops_a = self.estimate_query_cost(plan_a)
        cost_b, ops_b = self.estimate_query_cost(plan_b)
        
        return {
            'plan_a_cost': cost_a,
            'plan_b_cost': cost_b,
            'better_plan': 'A' if cost_a < cost_b else 'B',
            'cost_difference': abs(cost_a - cost_b),
            'improvement_ratio': max(cost_a, cost_b) / min(cost_a, cost_b) if min(cost_a, cost_b) > 0 else 1.0,
            'plan_a_operations': len(ops_a),
            'plan_b_operations': len(ops_b)
        }
    
    def get_cost_breakdown(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed cost breakdown for an execution plan.
        
        Args:
            execution_plan: Execution plan to analyze
            
        Returns:
            Detailed cost breakdown
        """
        total_cost, operation_costs = self.estimate_query_cost(execution_plan)
        
        # Aggregate costs by type
        cost_by_type = {}
        total_cpu = 0
        total_io = 0
        total_memory = 0
        
        for op_cost in operation_costs:
            op_type = op_cost.operation_type.value
            if op_type not in cost_by_type:
                cost_by_type[op_type] = []
            cost_by_type[op_type].append(op_cost.total_cost)
            
            total_cpu += op_cost.cpu_cost
            total_io += op_cost.io_cost
            total_memory += op_cost.memory_cost
        
        return {
            'total_cost': total_cost,
            'total_cpu_cost': total_cpu,
            'total_io_cost': total_io,
            'total_memory_cost': total_memory,
            'cost_by_operation_type': {
                op_type: {
                    'count': len(costs),
                    'total_cost': sum(costs),
                    'average_cost': sum(costs) / len(costs)
                }
                for op_type, costs in cost_by_type.items()
            },
            'operation_details': [
                {
                    'type': op.operation_type.value,
                    'cost': op.total_cost,
                    'cardinality': op.cardinality_estimate,
                    'cpu_cost': op.cpu_cost,
                    'io_cost': op.io_cost,
                    'memory_cost': op.memory_cost
                }
                for op in operation_costs
            ]
        }