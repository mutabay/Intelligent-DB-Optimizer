"""
Evaluation module for DQN-based query optimizer.

This module provides comprehensive evaluation and benchmarking functionality
for the multi-agent DQN system against baseline optimizers.
"""
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.agents.dqn_agent import MultiAgentDQN
from src.agents.rl_environment import QueryOptimizationEnv
from src.database_environment.db_simulator import DatabaseSimulator
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.optimization.rule_based_optimizer import RuleBasedOptimizer
from src.utils.logging import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    execution_time: float
    cost_estimate: float
    improvement_over_baseline: float
    optimization_choices: Dict[str, Any]
    query_complexity: int
    timestamp: str


class QueryBenchmark:
    """Handles query benchmark generation and execution."""
    
    def __init__(self, db_simulator: DatabaseSimulator):
        """
        Initialize benchmark.
        
        Args:
            db_simulator: Database simulator instance
        """
        self.db_simulator = db_simulator
        self.benchmark_queries = self._generate_benchmark_queries()
    
    def _generate_benchmark_queries(self) -> List[str]:
        """Generate diverse benchmark queries for evaluation."""
        queries = [
            # Simple selection queries
            "SELECT * FROM customers WHERE nation_key = 1",
            "SELECT * FROM orders WHERE order_date > '1995-01-01'",
            
            # Join queries of varying complexity
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
            WHERE o.order_date > '1995-01-01'
            GROUP BY c.customer_key, c.name
            HAVING SUM(o.total_price) > 10000
            """,
            
            # Complex multi-table joins
            """
            SELECT c.name, o.order_date, l.quantity, l.extended_price
            FROM customers c
            JOIN orders o ON c.customer_key = o.customer_key
            JOIN lineitem l ON o.order_key = l.order_key
            WHERE c.nation_key IN (1, 2, 3)
              AND o.order_date BETWEEN '1995-01-01' AND '1996-12-31'
              AND l.quantity > 10
            """,
            
            # Aggregation queries
            """
            SELECT o.order_priority, COUNT(*) as order_count, AVG(o.total_price) as avg_price
            FROM orders o
            JOIN lineitem l ON o.order_key = l.order_key
            WHERE l.ship_date > '1995-06-01'
            GROUP BY o.order_priority
            ORDER BY avg_price DESC
            """,
            
            # Subqueries
            """
            SELECT *
            FROM customers c
            WHERE c.customer_key IN (
                SELECT o.customer_key
                FROM orders o
                WHERE o.total_price > (
                    SELECT AVG(total_price) * 1.5
                    FROM orders
                    WHERE order_date > '1995-01-01'
                )
            )
            """
        ]
        
        return queries
    
    def execute_query(self, query: str, optimization_params: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Execute query and return performance metrics.
        
        Args:
            query: SQL query string
            optimization_params: Optimization parameters to apply
            
        Returns:
            Tuple of (execution_time, execution_info)
        """
        start_time = time.time()
        
        try:
            # Apply optimizations if provided
            if optimization_params:
                # This would apply actual optimizations in a real implementation
                # For simulation, we'll estimate the impact
                pass
            
            # Execute query
            result = self.db_simulator.execute_query(query)
            execution_time = time.time() - start_time
            
            execution_info = {
                'rows_returned': len(result) if result else 0,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_info = {
                'rows_returned': 0,
                'success': False,
                'error': str(e)
            }
            logger.error(f"Query execution failed: {e}")
        
        return execution_time, execution_info


class DQNEvaluator:
    """Comprehensive evaluation framework for DQN query optimizer."""
    
    def __init__(self, save_dir: str = "results/evaluation"):
        """
        Initialize evaluator.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.evaluation_results = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def evaluate_dqn_vs_baseline(self, env: QueryOptimizationEnv, dqn: MultiAgentDQN,
                                 baseline_optimizer: RuleBasedOptimizer,
                                 benchmark: QueryBenchmark, 
                                 num_trials: int = 10) -> Dict:
        """
        Compare DQN performance against baseline optimizer.
        
        Args:
            env: RL environment
            dqn: Trained DQN system
            baseline_optimizer: Rule-based baseline
            benchmark: Query benchmark suite
            num_trials: Number of evaluation trials per query
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting DQN vs Baseline evaluation with {num_trials} trials")
        
        results = {
            'dqn_results': [],
            'baseline_results': [],
            'comparisons': [],
            'summary_stats': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for query_idx, query in enumerate(benchmark.benchmark_queries):
            logger.info(f"Evaluating query {query_idx + 1}/{len(benchmark.benchmark_queries)}")
            
            # Evaluate DQN performance
            dqn_metrics = self._evaluate_dqn_on_query(env, dqn, query, num_trials)
            results['dqn_results'].append(dqn_metrics)
            
            # Evaluate baseline performance
            baseline_metrics = self._evaluate_baseline_on_query(
                baseline_optimizer, benchmark, query, num_trials
            )
            results['baseline_results'].append(baseline_metrics)
            
            # Compare results
            comparison = self._compare_query_results(dqn_metrics, baseline_metrics, query_idx)
            results['comparisons'].append(comparison)
        
        # Calculate summary statistics
        results['summary_stats'] = self._calculate_summary_stats(results['comparisons'])
        
        # Save results
        self._save_evaluation_results(results)
        
        logger.info("Evaluation completed successfully")
        return results
    
    def _evaluate_dqn_on_query(self, env: QueryOptimizationEnv, dqn: MultiAgentDQN,
                              query: str, num_trials: int) -> Dict:
        """Evaluate DQN performance on single query."""
        trial_results = []
        
        for trial in range(num_trials):
            # Reset environment with query
            state = env.reset(query=query)
            
            # Get DQN optimization decisions
            actions = dqn.get_actions(state, deterministic=True)  # Use deterministic for evaluation
            
            # Apply optimizations and measure performance
            next_state, reward, done, truncated, info = env.step(actions)
            
            metrics = EvaluationMetrics(
                execution_time=info.get('execution_time', 0.0),
                cost_estimate=info.get('cost_estimate', 0.0),
                improvement_over_baseline=info.get('improvement', 0.0),
                optimization_choices={
                    'join_ordering': actions[0],
                    'index_advice': actions[1],
                    'cache_strategy': actions[2],
                    'resource_allocation': actions[3]
                },
                query_complexity=info.get('complexity', 1),
                timestamp=datetime.now().isoformat()
            )
            
            trial_results.append(metrics)
        
        # Aggregate trial results
        return self._aggregate_trial_results(trial_results, 'dqn')
    
    def _evaluate_baseline_on_query(self, baseline_optimizer: RuleBasedOptimizer,
                                   benchmark: QueryBenchmark, query: str,
                                   num_trials: int) -> Dict:
        """Evaluate baseline optimizer performance on single query."""
        trial_results = []
        
        for trial in range(num_trials):
            # Get baseline optimization decisions
            optimization_params = baseline_optimizer.optimize_query(query)
            
            # Execute query with baseline optimizations
            execution_time, execution_info = benchmark.execute_query(query, optimization_params)
            
            metrics = EvaluationMetrics(
                execution_time=execution_time,
                cost_estimate=baseline_optimizer.estimate_cost(query),
                improvement_over_baseline=0.0,  # Baseline is the reference point
                optimization_choices=optimization_params,
                query_complexity=baseline_optimizer.analyze_complexity(query),
                timestamp=datetime.now().isoformat()
            )
            
            trial_results.append(metrics)
        
        return self._aggregate_trial_results(trial_results, 'baseline')
    
    def _aggregate_trial_results(self, trial_results: List[EvaluationMetrics], 
                                method: str) -> Dict:
        """Aggregate results from multiple trials."""
        execution_times = [r.execution_time for r in trial_results]
        cost_estimates = [r.cost_estimate for r in trial_results]
        improvements = [r.improvement_over_baseline for r in trial_results]
        
        return {
            'method': method,
            'num_trials': len(trial_results),
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'median': np.median(execution_times)
            },
            'cost_estimate': {
                'mean': np.mean(cost_estimates),
                'std': np.std(cost_estimates),
                'min': np.min(cost_estimates),
                'max': np.max(cost_estimates)
            },
            'improvement': {
                'mean': np.mean(improvements),
                'std': np.std(improvements)
            },
            'sample_optimization': trial_results[0].optimization_choices if trial_results else {},
            'query_complexity': trial_results[0].query_complexity if trial_results else 1
        }
    
    def _compare_query_results(self, dqn_results: Dict, baseline_results: Dict,
                              query_idx: int) -> Dict:
        """Compare DQN vs baseline results for single query."""
        dqn_time = dqn_results['execution_time']['mean']
        baseline_time = baseline_results['execution_time']['mean']
        
        speedup = baseline_time / dqn_time if dqn_time > 0 else 1.0
        improvement_pct = ((baseline_time - dqn_time) / baseline_time * 100) if baseline_time > 0 else 0.0
        
        return {
            'query_index': query_idx,
            'dqn_execution_time': dqn_time,
            'baseline_execution_time': baseline_time,
            'speedup': speedup,
            'improvement_percentage': improvement_pct,
            'dqn_cost': dqn_results['cost_estimate']['mean'],
            'baseline_cost': baseline_results['cost_estimate']['mean'],
            'dqn_wins': dqn_time < baseline_time,
            'statistical_significance': self._check_significance(dqn_results, baseline_results)
        }
    
    def _check_significance(self, dqn_results: Dict, baseline_results: Dict) -> Dict:
        """Check statistical significance of performance difference."""
        # Simple significance test based on non-overlapping confidence intervals
        dqn_mean = dqn_results['execution_time']['mean']
        dqn_std = dqn_results['execution_time']['std']
        baseline_mean = baseline_results['execution_time']['mean']
        baseline_std = baseline_results['execution_time']['std']
        
        # 95% confidence intervals (assuming normal distribution)
        dqn_ci = (dqn_mean - 1.96 * dqn_std, dqn_mean + 1.96 * dqn_std)
        baseline_ci = (baseline_mean - 1.96 * baseline_std, baseline_mean + 1.96 * baseline_std)
        
        # Check if confidence intervals overlap
        no_overlap = dqn_ci[1] < baseline_ci[0] or baseline_ci[1] < dqn_ci[0]
        
        return {
            'significant': no_overlap,
            'dqn_confidence_interval': dqn_ci,
            'baseline_confidence_interval': baseline_ci,
            'effect_size': abs(dqn_mean - baseline_mean) / max(dqn_std, baseline_std) if max(dqn_std, baseline_std) > 0 else 0
        }
    
    def _calculate_summary_stats(self, comparisons: List[Dict]) -> Dict:
        """Calculate overall summary statistics."""
        total_queries = len(comparisons)
        dqn_wins = sum(1 for c in comparisons if c['dqn_wins'])
        
        speedups = [c['speedup'] for c in comparisons]
        improvements = [c['improvement_percentage'] for c in comparisons]
        
        return {
            'total_queries_evaluated': total_queries,
            'dqn_wins': dqn_wins,
            'baseline_wins': total_queries - dqn_wins,
            'win_rate': dqn_wins / total_queries if total_queries > 0 else 0.0,
            'average_speedup': np.mean(speedups),
            'median_speedup': np.median(speedups),
            'average_improvement_pct': np.mean(improvements),
            'median_improvement_pct': np.median(improvements),
            'best_speedup': np.max(speedups),
            'worst_speedup': np.min(speedups),
            'significant_improvements': sum(1 for c in comparisons if c['statistical_significance']['significant'] and c['dqn_wins'])
        }
    
    def _save_evaluation_results(self, results: Dict) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.save_dir, f"dqn_evaluation_{timestamp}.json")
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all numpy types
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_results = deep_convert(results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
    
    def plot_evaluation_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Generate comprehensive evaluation visualization."""
        comparisons = results['comparisons']
        
        if not comparisons:
            logger.warning("No comparison data available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN vs Baseline Optimization Evaluation', fontsize=16)
        
        # Query performance comparison
        query_indices = [c['query_index'] for c in comparisons]
        dqn_times = [c['dqn_execution_time'] for c in comparisons]
        baseline_times = [c['baseline_execution_time'] for c in comparisons]
        
        ax1.bar([i - 0.2 for i in query_indices], dqn_times, 0.4, label='DQN', alpha=0.8)
        ax1.bar([i + 0.2 for i in query_indices], baseline_times, 0.4, label='Baseline', alpha=0.8)
        ax1.set_xlabel('Query Index')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup distribution
        speedups = [c['speedup'] for c in comparisons]
        ax2.hist(speedups, bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', label='No Improvement')
        ax2.axvline(x=np.mean(speedups), color='green', linestyle='-', label=f'Mean: {np.mean(speedups):.2f}x')
        ax2.set_xlabel('Speedup Factor')
        ax2.set_ylabel('Number of Queries')
        ax2.set_title('Speedup Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Improvement percentages
        improvements = [c['improvement_percentage'] for c in comparisons]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax3.bar(query_indices, improvements, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax3.set_xlabel('Query Index')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Performance Improvement by Query')
        ax3.grid(True, alpha=0.3)
        
        # Win rate and summary stats
        summary = results['summary_stats']
        categories = ['DQN Wins', 'Baseline Wins']
        values = [summary['dqn_wins'], summary['baseline_wins']]
        colors = ['green', 'red']
        
        ax4.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f"Win Rate\n(Avg Speedup: {summary['average_speedup']:.2f}x)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate human-readable evaluation report."""
        summary = results['summary_stats']
        
        report = f"""
# DQN Query Optimizer Evaluation Report

**Evaluation Date**: {results['timestamp']}
**Total Queries Evaluated**: {summary['total_queries_evaluated']}

## Overall Performance

- **DQN Wins**: {summary['dqn_wins']} ({summary['win_rate']:.1%})
- **Baseline Wins**: {summary['baseline_wins']} ({1-summary['win_rate']:.1%})
- **Average Speedup**: {summary['average_speedup']:.2f}x
- **Median Speedup**: {summary['median_speedup']:.2f}x
- **Average Improvement**: {summary['average_improvement_pct']:.1f}%

## Performance Range

- **Best Speedup**: {summary['best_speedup']:.2f}x
- **Worst Performance**: {summary['worst_speedup']:.2f}x
- **Statistically Significant Improvements**: {summary['significant_improvements']}/{summary['total_queries_evaluated']}

## Per-Query Analysis

"""
        
        for i, comparison in enumerate(results['comparisons']):
            status = "DQN Win" if comparison['dqn_wins'] else "❌ Baseline Win"
            sig = "Significant" if comparison['statistical_significance']['significant'] else ""
            
            report += f"""
**Query {i+1}**: {status} {sig}
- DQN Time: {comparison['dqn_execution_time']:.3f}s
- Baseline Time: {comparison['baseline_execution_time']:.3f}s  
- Speedup: {comparison['speedup']:.2f}x
- Improvement: {comparison['improvement_percentage']:.1f}%

"""
        
        return report