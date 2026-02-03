"""
Performance Benchmarking Suite for Intelligent Database Query Optimizer

This module provides comprehensive performance testing and benchmarking
capabilities to evaluate the hybrid AI optimization system against
traditional approaches and measure system performance characteristics.
"""

import os
import sys
import pytest
import time
import tempfile
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import psutil
import gc

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import IntelligentDBOptimizer
from src.utils.logging import logger


class PerformanceBenchmark:
    """
    Performance benchmarking suite for the intelligent database optimizer.
    """

    def __init__(self, db_path: str = None):
        """Initialize performance benchmark suite."""
        self.db_path = db_path or ":memory:"
        self.system = None
        self.results = {
            'optimization_times': {},
            'cost_improvements': {},
            'memory_usage': {},
            'accuracy_metrics': {}
        }

    def setup_system(self):
        """Setup system for performance testing."""
        logger.info("Setting up performance testing environment...")
        self.system = IntelligentDBOptimizer(db_type="sqlite", db_path=self.db_path)
        success = self.system.initialize_system()
        assert success, "Failed to initialize system for performance testing"
        logger.info("✅ Performance testing environment ready")

    def teardown_system(self):
        """Clean up system after performance testing."""
        if self.system:
            self.system.cleanup()
            self.system = None

    def benchmark_optimization_strategies(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Benchmark different optimization strategies against test queries.
        
        Returns performance metrics comparing rule-based, DQN-based, and hybrid approaches.
        """
        logger.info("🚀 Starting optimization strategy benchmark...")
        
        strategies = ["rule_based", "dqn_based", "hybrid"]
        strategy_results = {strategy: [] for strategy in strategies}
        
        for query_idx, query in enumerate(test_queries):
            logger.info(f"Testing query {query_idx + 1}/{len(test_queries)}")
            
            for strategy in strategies:
                try:
                    # Measure optimization time
                    start_time = time.perf_counter()
                    result = self.system.optimize_query(query, strategy=strategy)
                    end_time = time.perf_counter()
                    
                    optimization_time = end_time - start_time
                    estimated_cost = result.get('estimated_cost', 0)
                    
                    strategy_results[strategy].append({
                        'query_id': query_idx,
                        'optimization_time': optimization_time,
                        'estimated_cost': estimated_cost,
                        'execution_time': result.get('execution_time', 0),
                        'success': True
                    })
                    
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed on query {query_idx}: {e}")
                    strategy_results[strategy].append({
                        'query_id': query_idx,
                        'optimization_time': float('inf'),
                        'estimated_cost': float('inf'),
                        'execution_time': float('inf'),
                        'success': False
                    })

        # Calculate performance metrics
        metrics = self._calculate_strategy_metrics(strategy_results)
        self.results['optimization_times'] = metrics
        
        logger.info("✅ Optimization strategy benchmark completed")
        return metrics

    def benchmark_scalability(self, query_sizes: List[int]) -> Dict[str, Any]:
        """
        Benchmark system scalability with queries of different complexity levels.
        
        Tests how optimization time scales with query complexity.
        """
        logger.info("📈 Starting scalability benchmark...")
        
        scalability_results = []
        
        for size in query_sizes:
            # Generate query of specified complexity
            query = self._generate_complex_query(size)
            
            # Measure optimization performance
            start_time = time.perf_counter()
            memory_before = self._get_memory_usage()
            
            try:
                result = self.system.optimize_query(query, strategy="hybrid")
                success = True
                cost = result.get('estimated_cost', 0)
            except Exception as e:
                logger.warning(f"Scalability test failed for size {size}: {e}")
                success = False
                cost = float('inf')
            
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()
            
            scalability_results.append({
                'query_complexity': size,
                'optimization_time': end_time - start_time,
                'memory_delta': memory_after - memory_before,
                'estimated_cost': cost,
                'success': success
            })
            
            logger.info(f"Complexity {size}: {end_time - start_time:.3f}s, Memory: +{memory_after - memory_before:.1f}MB")

        self.results['scalability'] = scalability_results
        logger.info("✅ Scalability benchmark completed")
        return scalability_results

    def benchmark_accuracy(self, ground_truth_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark optimization accuracy against known ground truth results.
        
        Args:
            ground_truth_queries: List of queries with known optimal costs/plans
        """
        logger.info("🎯 Starting accuracy benchmark...")
        
        accuracy_results = []
        
        for query_data in ground_truth_queries:
            query = query_data['query']
            ground_truth_cost = query_data['optimal_cost']
            
            # Test each strategy's accuracy
            strategy_accuracy = {}
            
            for strategy in ["rule_based", "dqn_based", "hybrid"]:
                try:
                    result = self.system.optimize_query(query, strategy=strategy)
                    predicted_cost = result.get('estimated_cost', float('inf'))
                    
                    # Calculate accuracy metrics
                    relative_error = abs(predicted_cost - ground_truth_cost) / ground_truth_cost
                    strategy_accuracy[strategy] = {
                        'predicted_cost': predicted_cost,
                        'relative_error': relative_error,
                        'accuracy_score': max(0, 1 - relative_error)  # 1 = perfect, 0 = terrible
                    }
                    
                except Exception as e:
                    logger.warning(f"Accuracy test failed for {strategy}: {e}")
                    strategy_accuracy[strategy] = {
                        'predicted_cost': float('inf'),
                        'relative_error': float('inf'),
                        'accuracy_score': 0
                    }
            
            accuracy_results.append({
                'query_id': query_data.get('id', len(accuracy_results)),
                'ground_truth_cost': ground_truth_cost,
                'strategy_results': strategy_accuracy
            })

        # Calculate overall accuracy metrics
        overall_accuracy = self._calculate_accuracy_metrics(accuracy_results)
        self.results['accuracy_metrics'] = overall_accuracy
        
        logger.info("✅ Accuracy benchmark completed")
        return overall_accuracy

    def benchmark_memory_efficiency(self, test_duration: int = 60) -> Dict[str, Any]:
        """
        Benchmark memory efficiency over sustained operation.
        
        Args:
            test_duration: Duration of test in seconds
        """
        logger.info(f"💾 Starting {test_duration}s memory efficiency benchmark...")
        
        memory_samples = []
        start_time = time.time()
        
        # Simple test queries for sustained operation
        test_queries = [
            "SELECT * FROM customers WHERE nation_key = 1",
            "SELECT c.name, o.order_date FROM customers c JOIN orders o ON c.customer_key = o.customer_key LIMIT 10",
            "SELECT COUNT(*) FROM lineitem WHERE quantity > 10"
        ]
        
        query_count = 0
        while time.time() - start_time < test_duration:
            # Perform optimization
            query = test_queries[query_count % len(test_queries)]
            try:
                self.system.optimize_query(query, strategy="hybrid")
                query_count += 1
            except:
                pass  # Continue on errors
            
            # Sample memory usage
            memory_usage = self._get_memory_usage()
            memory_samples.append({
                'timestamp': time.time() - start_time,
                'memory_mb': memory_usage,
                'queries_processed': query_count
            })
            
            time.sleep(1.0)  # Sample every second

        # Analyze memory trends
        memory_metrics = {
            'initial_memory': memory_samples[0]['memory_mb'],
            'final_memory': memory_samples[-1]['memory_mb'],
            'peak_memory': max(sample['memory_mb'] for sample in memory_samples),
            'average_memory': statistics.mean(sample['memory_mb'] for sample in memory_samples),
            'memory_growth': memory_samples[-1]['memory_mb'] - memory_samples[0]['memory_mb'],
            'queries_per_second': query_count / test_duration,
            'total_queries': query_count,
            'samples': memory_samples
        }
        
        self.results['memory_usage'] = memory_metrics
        logger.info(f"✅ Memory benchmark: {memory_metrics['memory_growth']:.1f}MB growth over {query_count} queries")
        return memory_metrics

    def benchmark_concurrent_performance(self, num_threads: int = 4, queries_per_thread: int = 10) -> Dict[str, Any]:
        """
        Benchmark system performance under concurrent load.
        
        Args:
            num_threads: Number of concurrent optimization threads
            queries_per_thread: Number of queries each thread processes
        """
        import threading
        import queue
        
        logger.info(f"⚡ Starting concurrent performance benchmark: {num_threads} threads, {queries_per_thread} queries each")
        
        results_queue = queue.Queue()
        threads = []
        
        def worker_function(thread_id: int):
            """Worker function for concurrent optimization testing."""
            thread_results = []
            
            for i in range(queries_per_thread):
                query = f"SELECT * FROM customers WHERE customer_key = {thread_id * queries_per_thread + i + 1}"
                
                start_time = time.perf_counter()
                try:
                    result = self.system.optimize_query(query, strategy="rule_based")
                    success = True
                    cost = result.get('estimated_cost', 0)
                except Exception as e:
                    success = False
                    cost = float('inf')
                end_time = time.perf_counter()
                
                thread_results.append({
                    'thread_id': thread_id,
                    'query_id': i,
                    'optimization_time': end_time - start_time,
                    'estimated_cost': cost,
                    'success': success
                })
            
            results_queue.put(thread_results)

        # Launch worker threads
        start_time = time.perf_counter()
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_function, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()
        end_time = time.perf_counter()

        # Collect results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)

        # Calculate concurrent performance metrics
        successful_queries = [r for r in all_results if r['success']]
        total_time = end_time - start_time
        
        concurrent_metrics = {
            'total_queries': len(all_results),
            'successful_queries': len(successful_queries),
            'success_rate': len(successful_queries) / len(all_results) if all_results else 0,
            'total_execution_time': total_time,
            'queries_per_second': len(all_results) / total_time if total_time > 0 else 0,
            'average_optimization_time': statistics.mean(r['optimization_time'] for r in successful_queries) if successful_queries else 0,
            'max_optimization_time': max(r['optimization_time'] for r in successful_queries) if successful_queries else 0,
            'thread_results': all_results
        }
        
        self.results['concurrent_performance'] = concurrent_metrics
        logger.info(f"✅ Concurrent benchmark: {concurrent_metrics['success_rate']:.1%} success rate, {concurrent_metrics['queries_per_second']:.1f} QPS")
        return concurrent_metrics

    def generate_performance_report(self, output_dir: str = "results/performance_charts") -> str:
        """
        Generate comprehensive performance report with visualizations.
        
        Returns path to generated report.
        """
        logger.info("📊 Generating performance report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        self._plot_optimization_times(output_dir)
        self._plot_memory_usage(output_dir)
        self._plot_scalability(output_dir)
        self._plot_accuracy_comparison(output_dir)
        
        # Generate summary report
        report_path = os.path.join(output_dir, "performance_report.md")
        self._generate_markdown_report(report_path)
        
        logger.info(f"✅ Performance report generated: {report_path}")
        return report_path

    def _generate_complex_query(self, complexity: int) -> str:
        """Generate SQL query with specified complexity level."""
        base_query = "SELECT c.name"
        tables = ["customers c"]
        joins = []
        where_conditions = []
        
        # Add complexity through joins and conditions
        if complexity >= 2:
            base_query += ", o.order_date"
            tables.append("orders o")
            joins.append("c.customer_key = o.customer_key")
        
        if complexity >= 3:
            base_query += ", l.quantity"
            tables.append("lineitem l")
            joins.append("o.order_key = l.order_key")
        
        # Add WHERE conditions based on complexity
        for i in range(min(complexity, 3)):
            if i == 0:
                where_conditions.append("c.nation_key IN (1, 2, 3)")
            elif i == 1:
                where_conditions.append("o.order_date > '1995-01-01'")
            elif i == 2:
                where_conditions.append("l.quantity > 10")
        
        # Construct final query
        query = f"{base_query} FROM {', '.join(tables)}"
        
        if joins:
            for i, join in enumerate(joins):
                if i == 0:
                    query += f" WHERE {join}"
                else:
                    query += f" AND {join}"
        
        if where_conditions:
            if not joins:
                query += " WHERE "
            else:
                query += " AND "
            query += " AND ".join(where_conditions)
        
        if complexity >= 4:
            query += " GROUP BY c.customer_key, c.name"
            if complexity >= 5:
                query += " HAVING COUNT(o.order_key) > 1"
        
        query += " LIMIT 100"
        
        return query

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _calculate_strategy_metrics(self, strategy_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate performance metrics for optimization strategies."""
        metrics = {}
        
        for strategy, results in strategy_results.items():
            successful_results = [r for r in results if r['success']]
            
            if successful_results:
                metrics[strategy] = {
                    'success_rate': len(successful_results) / len(results),
                    'avg_optimization_time': statistics.mean(r['optimization_time'] for r in successful_results),
                    'median_optimization_time': statistics.median(r['optimization_time'] for r in successful_results),
                    'avg_estimated_cost': statistics.mean(r['estimated_cost'] for r in successful_results),
                    'min_optimization_time': min(r['optimization_time'] for r in successful_results),
                    'max_optimization_time': max(r['optimization_time'] for r in successful_results)
                }
            else:
                metrics[strategy] = {
                    'success_rate': 0,
                    'avg_optimization_time': float('inf'),
                    'median_optimization_time': float('inf'),
                    'avg_estimated_cost': float('inf'),
                    'min_optimization_time': float('inf'),
                    'max_optimization_time': float('inf')
                }
        
        return metrics

    def _calculate_accuracy_metrics(self, accuracy_results: List[Dict]) -> Dict[str, Any]:
        """Calculate accuracy metrics across all test queries."""
        strategy_accuracy = {'rule_based': [], 'dqn_based': [], 'hybrid': []}
        
        for result in accuracy_results:
            for strategy, strategy_result in result['strategy_results'].items():
                strategy_accuracy[strategy].append(strategy_result['accuracy_score'])
        
        overall_metrics = {}
        for strategy, scores in strategy_accuracy.items():
            if scores:
                overall_metrics[strategy] = {
                    'average_accuracy': statistics.mean(scores),
                    'median_accuracy': statistics.median(scores),
                    'min_accuracy': min(scores),
                    'max_accuracy': max(scores),
                    'accuracy_std': statistics.stdev(scores) if len(scores) > 1 else 0
                }
            else:
                overall_metrics[strategy] = {
                    'average_accuracy': 0,
                    'median_accuracy': 0,
                    'min_accuracy': 0,
                    'max_accuracy': 0,
                    'accuracy_std': 0
                }
        
        return overall_metrics

    def _plot_optimization_times(self, output_dir: str):
        """Generate optimization time comparison plots."""
        if 'optimization_times' not in self.results:
            return
        
        strategies = list(self.results['optimization_times'].keys())
        avg_times = [self.results['optimization_times'][s]['avg_optimization_time'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        plt.bar(strategies, avg_times, color=['blue', 'orange', 'green'])
        plt.title('Average Optimization Time by Strategy')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Optimization Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_times.png'))
        plt.close()

    def _plot_memory_usage(self, output_dir: str):
        """Generate memory usage plots."""
        if 'memory_usage' not in self.results or 'samples' not in self.results['memory_usage']:
            return
        
        samples = self.results['memory_usage']['samples']
        timestamps = [s['timestamp'] for s in samples]
        memory_values = [s['memory_mb'] for s in samples]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_values, 'b-', linewidth=2)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
        plt.close()

    def _plot_scalability(self, output_dir: str):
        """Generate scalability plots."""
        if 'scalability' not in self.results:
            return
        
        scalability_data = self.results['scalability']
        complexities = [r['query_complexity'] for r in scalability_data if r['success']]
        opt_times = [r['optimization_time'] for r in scalability_data if r['success']]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(complexities, opt_times, color='red', alpha=0.7)
        plt.plot(complexities, opt_times, 'r--', alpha=0.5)
        plt.title('Optimization Time vs Query Complexity')
        plt.xlabel('Query Complexity')
        plt.ylabel('Optimization Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scalability.png'))
        plt.close()

    def _plot_accuracy_comparison(self, output_dir: str):
        """Generate accuracy comparison plots."""
        if 'accuracy_metrics' not in self.results:
            return
        
        strategies = list(self.results['accuracy_metrics'].keys())
        accuracies = [self.results['accuracy_metrics'][s]['average_accuracy'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        plt.bar(strategies, accuracies, color=['cyan', 'magenta', 'yellow'])
        plt.title('Average Accuracy by Strategy')
        plt.ylabel('Accuracy Score (0-1)')
        plt.xlabel('Optimization Strategy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()

    def _generate_markdown_report(self, report_path: str):
        """Generate markdown performance report."""
        with open(report_path, 'w') as f:
            f.write("# Performance Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Optimization Times Section
            if 'optimization_times' in self.results:
                f.write("## Optimization Strategy Performance\n\n")
                f.write("| Strategy | Success Rate | Avg Time (s) | Median Time (s) | Min Time (s) | Max Time (s) |\n")
                f.write("|----------|--------------|--------------|-----------------|--------------|-------------|\n")
                
                for strategy, metrics in self.results['optimization_times'].items():
                    f.write(f"| {strategy} | {metrics['success_rate']:.1%} | {metrics['avg_optimization_time']:.3f} | "
                           f"{metrics['median_optimization_time']:.3f} | {metrics['min_optimization_time']:.3f} | "
                           f"{metrics['max_optimization_time']:.3f} |\n")
                f.write("\n")
            
            # Memory Usage Section
            if 'memory_usage' in self.results:
                mem = self.results['memory_usage']
                f.write("## Memory Efficiency\n\n")
                f.write(f"- Initial Memory: {mem['initial_memory']:.1f} MB\n")
                f.write(f"- Final Memory: {mem['final_memory']:.1f} MB\n")
                f.write(f"- Peak Memory: {mem['peak_memory']:.1f} MB\n")
                f.write(f"- Memory Growth: {mem['memory_growth']:.1f} MB\n")
                f.write(f"- Queries Per Second: {mem['queries_per_second']:.1f}\n\n")
            
            # Scalability Section
            if 'scalability' in self.results:
                f.write("## Scalability Analysis\n\n")
                f.write("| Complexity | Optimization Time (s) | Memory Delta (MB) | Success |\n")
                f.write("|------------|---------------------|-------------------|----------|\n")
                
                for result in self.results['scalability']:
                    f.write(f"| {result['query_complexity']} | {result['optimization_time']:.3f} | "
                           f"{result['memory_delta']:.1f} | {'✅' if result['success'] else '❌'} |\n")
                f.write("\n")
            
            # Accuracy Section
            if 'accuracy_metrics' in self.results:
                f.write("## Accuracy Metrics\n\n")
                f.write("| Strategy | Avg Accuracy | Median Accuracy | Min Accuracy | Max Accuracy |\n")
                f.write("|----------|--------------|-----------------|--------------|-------------|\n")
                
                for strategy, metrics in self.results['accuracy_metrics'].items():
                    f.write(f"| {strategy} | {metrics['average_accuracy']:.3f} | {metrics['median_accuracy']:.3f} | "
                           f"{metrics['min_accuracy']:.3f} | {metrics['max_accuracy']:.3f} |\n")
                f.write("\n")


class TestPerformanceBenchmarks:
    """
    Pytest-based performance tests.
    """

    @pytest.fixture(scope="class")
    def benchmark_suite(self):
        """Initialize benchmark suite for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        suite = PerformanceBenchmark(db_path=temp_db)
        suite.setup_system()
        yield suite
        suite.teardown_system()
        os.unlink(temp_db)

    def test_optimization_strategy_benchmark(self, benchmark_suite):
        """Test optimization strategy benchmarking."""
        test_queries = [
            "SELECT * FROM customers WHERE nation_key = 1",
            "SELECT c.name, o.order_date FROM customers c JOIN orders o ON c.customer_key = o.customer_key LIMIT 10"
        ]
        
        metrics = benchmark_suite.benchmark_optimization_strategies(test_queries)
        
        # Verify metrics structure
        assert 'rule_based' in metrics, "Missing rule-based metrics"
        assert 'hybrid' in metrics, "Missing hybrid metrics"
        
        for strategy, strategy_metrics in metrics.items():
            assert 'success_rate' in strategy_metrics, f"Missing success rate for {strategy}"
            assert 'avg_optimization_time' in strategy_metrics, f"Missing avg time for {strategy}"
            assert strategy_metrics['success_rate'] >= 0, f"Invalid success rate for {strategy}"
        
        logger.info("✅ Optimization strategy benchmark test passed")

    def test_scalability_benchmark(self, benchmark_suite):
        """Test scalability benchmarking."""
        query_sizes = [1, 2, 3]  # Small sizes for testing
        
        results = benchmark_suite.benchmark_scalability(query_sizes)
        
        assert len(results) == len(query_sizes), "Incorrect number of scalability results"
        
        for result in results:
            assert 'query_complexity' in result, "Missing complexity in scalability result"
            assert 'optimization_time' in result, "Missing time in scalability result"
            assert 'success' in result, "Missing success flag in scalability result"
        
        logger.info("✅ Scalability benchmark test passed")

    def test_memory_efficiency_benchmark(self, benchmark_suite):
        """Test memory efficiency benchmarking."""
        metrics = benchmark_suite.benchmark_memory_efficiency(test_duration=10)  # Short test
        
        assert 'initial_memory' in metrics, "Missing initial memory metric"
        assert 'final_memory' in metrics, "Missing final memory metric"
        assert 'memory_growth' in metrics, "Missing memory growth metric"
        assert 'queries_per_second' in metrics, "Missing QPS metric"
        
        # Memory should not grow excessively
        assert metrics['memory_growth'] < 100, "Excessive memory growth detected"
        
        logger.info("✅ Memory efficiency benchmark test passed")

    def test_concurrent_performance_benchmark(self, benchmark_suite):
        """Test concurrent performance benchmarking."""
        metrics = benchmark_suite.benchmark_concurrent_performance(num_threads=2, queries_per_thread=3)
        
        assert 'total_queries' in metrics, "Missing total queries metric"
        assert 'success_rate' in metrics, "Missing success rate metric"
        assert 'queries_per_second' in metrics, "Missing QPS metric"
        
        assert metrics['success_rate'] > 0.5, "Low success rate in concurrent test"
        assert metrics['total_queries'] == 6, "Incorrect total query count"
        
        logger.info("✅ Concurrent performance benchmark test passed")

    def test_performance_report_generation(self, benchmark_suite):
        """Test performance report generation."""
        # Run small benchmarks to generate data
        test_queries = ["SELECT * FROM customers LIMIT 10"]
        benchmark_suite.benchmark_optimization_strategies(test_queries)
        
        # Generate report
        report_path = benchmark_suite.generate_performance_report()
        
        assert os.path.exists(report_path), "Performance report not generated"
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert "Performance Benchmark Report" in content, "Invalid report format"
            assert "Optimization Strategy Performance" in content, "Missing strategy section"
        
        logger.info("✅ Performance report generation test passed")


if __name__ == "__main__":
    # Run standalone performance benchmark
    benchmark = PerformanceBenchmark()
    benchmark.setup_system()
    
    try:
        # Quick benchmark suite
        test_queries = [
            "SELECT * FROM customers WHERE nation_key = 1",
            "SELECT c.name, o.order_date FROM customers c JOIN orders o ON c.customer_key = o.customer_key LIMIT 10",
            "SELECT COUNT(*) FROM lineitem WHERE quantity > 5"
        ]
        
        logger.info("🚀 Starting performance benchmark suite...")
        benchmark.benchmark_optimization_strategies(test_queries)
        benchmark.benchmark_scalability([1, 2, 3])
        benchmark.benchmark_memory_efficiency(30)
        benchmark.benchmark_concurrent_performance(3, 5)
        
        # Generate comprehensive report
        report_path = benchmark.generate_performance_report()
        logger.info(f"📊 Performance report generated: {report_path}")
        
    finally:
        benchmark.teardown_system()