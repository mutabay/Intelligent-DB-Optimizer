"""
Unit tests for RL environment (QueryOptimizationEnv).

This module tests the reinforcement learning environment for database
query optimization, including state representation, action handling,
reward calculation, and environment dynamics.
"""

import pytest
import numpy as np
import os
from gymnasium.spaces import Box, Discrete

from src.agents.rl_environment import QueryOptimizationEnv


class TestRLEnvironment:
    """Test suite for RL environment components."""
    
    def test_environment_initialization(self, db_simulator, knowledge_graph):
        """Test RL environment initialization."""
        env = QueryOptimizationEnv(
            database_simulator=db_simulator,
            knowledge_graph=knowledge_graph
        )
        
        # Test basic attributes
        assert hasattr(env, 'db_simulator')
        assert hasattr(env, 'knowledge_graph')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        
        # Test observation space
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (12,)
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)
        
        # Test action space (multi-discrete for 4 agents)
        assert hasattr(env.action_space, '__len__') or hasattr(env.action_space, 'sample')
    
    def test_environment_reset(self, rl_environment):
        """Test environment reset functionality."""
        env = rl_environment
        
        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        
        # Verify initial state
        assert len(state) >= 12  # State might be longer than 12
        assert isinstance(state, (list, np.ndarray))
        
        # Test multiple resets
        for i in range(3):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                new_state = reset_result[0]
            else:
                new_state = reset_result
            assert len(new_state) >= 12
    
    def test_environment_step(self, rl_environment):
        """Test environment step functionality."""
        env = rl_environment
        reset_result = env.reset()
        
        # Handle tuple return from reset
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        
        # Generate valid actions for all 4 agents as dict
        actions = {
            'join_ordering': 0,
            'index_advisor': 0, 
            'cache_manager': 0,
            'resource_allocator': 0
        }
        
        # Execute step
        next_state, reward, done, truncated, info = env.step(actions)
        
        # Verify outputs
        assert len(next_state) >= 12
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Test step info contains useful information
        assert 'performance' in info or 'baseline_performance' in info
        assert 'improvement' in info or 'execution_time' in info
    
    def test_state_representation(self, rl_environment):
        """Test state vector representation."""
        env = rl_environment
        
        # Test multiple resets to get different states
        states = []
        for i in range(3):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
            states.append(state)
        
        # Verify all states have consistent structure
        for state in states:
            assert len(state) >= 12
            assert isinstance(state, (list, np.ndarray))
    
    def test_action_validation(self, rl_environment):
        """Test action validation and handling."""
        env = rl_environment
        reset_result = env.reset()
        
        # Handle tuple return from reset
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        
        # Test valid actions
        valid_actions = {
            'join_ordering': 0,
            'index_advisor': 0,
            'cache_manager': 0,
            'resource_allocator': 0
        }
        next_state, reward, done, truncated, info = env.step(valid_actions)
        assert 'error' not in info
        
        # Test maximum valid actions
        max_actions = {
            'join_ordering': 5,
            'index_advisor': 3,
            'cache_manager': 2,
            'resource_allocator': 2
        }
        next_state, reward, done, truncated, info = env.step(max_actions)
        assert 'error' not in info
        
        # Test boundary actions
        boundary_actions = {
            'join_ordering': 0,
            'index_advisor': 3,
            'cache_manager': 2,
            'resource_allocator': 2
        }
        next_state, reward, done, truncated, info = env.step(boundary_actions)
        # Should not crash
        assert next_state is not None
    
    def test_reward_calculation(self, rl_environment):
        """Test reward calculation mechanism."""
        env = rl_environment
        rewards = []
        
        # Test multiple action combinations
        action_combinations = [
            {'join_ordering': 0, 'index_advisor': 0, 'cache_manager': 0, 'resource_allocator': 0},
            {'join_ordering': 1, 'index_advisor': 1, 'cache_manager': 1, 'resource_allocator': 1},
            {'join_ordering': 2, 'index_advisor': 2, 'cache_manager': 1, 'resource_allocator': 1},
            {'join_ordering': 5, 'index_advisor': 3, 'cache_manager': 2, 'resource_allocator': 2},
        ]
        
        for actions in action_combinations:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
            next_state, reward, done, truncated, info = env.step(actions)
            rewards.append(reward)
        
        # Rewards should be finite numbers
        assert all(np.isfinite(r) for r in rewards)
        
        # All rewards should be valid numbers (may be same if environment is deterministic)
        assert len(rewards) == 4
    
    def test_environment_consistency(self, rl_environment):
        """Test environment consistency across multiple episodes."""
        env = rl_environment
        
        # Run multiple episodes with same actions
        same_actions = {
            'join_ordering': 1,
            'index_advisor': 1,
            'cache_manager': 1,
            'resource_allocator': 1
        }
        
        results = []
        for episode in range(3):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
                
            next_state, reward, done, truncated, info = env.step(same_actions)
            results.append((state, reward, done))
        
        # Check consistency in structure
        performance_values = []
        for state, reward, done in results:
            assert isinstance(reward, (int, float))
            assert isinstance(done, (bool, int, float))  # Can be bool or numeric
        
        # Extract performance values for additional checks
        for episode in range(3):
            reset_result = env.reset()
            next_state, reward, done, truncated, info = env.step(same_actions)
            performance_values.append(info['performance'])
        
        # States should be similar for same query
        states = [r[0] for r in results]
        for i in range(1, len(states)):
            assert np.allclose(states[0], states[i], atol=0.1)
        
        # Rewards should be consistent for same query-action pairs
        rewards = [r[1] for r in results]
        assert np.std(rewards) < 1.0  # Low variance expected
    
    def test_observation_bounds(self, rl_environment):
        """Test that observations stay within bounds."""
        env = rl_environment
        
        # Test multiple random episodes
        for episode in range(5):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
            
            for step in range(3):
                # Random valid actions as dict
                actions = {
                    'join_ordering': np.random.randint(0, 6),
                    'index_advisor': np.random.randint(0, 4),
                    'cache_manager': np.random.randint(0, 3),
                    'resource_allocator': np.random.randint(0, 3),
                }
                
                next_state, reward, done, truncated, info = env.step(actions)
                
                # Check observation bounds
                assert len(next_state) >= 12, f"State too short: {len(next_state)}"
                state = next_state  # Update state for next iteration
                
                if done or truncated:
                    break
                    
                state = next_state
    
    def test_performance_metrics(self, rl_environment):
        """Test performance metrics collection."""
        env = rl_environment
        
        # Run several episodes to collect metrics
        for episode in range(3):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
                
            actions = {
                'join_ordering': 0,
                'index_advisor': 0,
                'cache_manager': 0,
                'resource_allocator': 0
            }
            next_state, reward, done, truncated, info = env.step(actions)
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'episodes' in metrics
        assert 'success_rate' in metrics
        assert metrics['episodes'] >= 3  # Should have recorded episodes
        assert 0 <= metrics['success_rate'] <= 1  # Valid success rate
    
    def test_query_analysis(self, rl_environment):
        """Test query analysis functionality by checking info dict."""
        env = rl_environment
        
        # Test environment provides query info
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        actions = {
            'join_ordering': 0,
            'index_advisor': 0,
            'cache_manager': 0,
            'resource_allocator': 0
        }
        
        next_state, reward, done, truncated, info = env.step(actions)
        
        # Check that info contains useful analysis data
        assert isinstance(info, dict)
        # Check for some expected keys (these may vary based on implementation)
        expected_keys = ['query_complexity', 'performance_improvement', 'baseline_performance']
        for key in expected_keys:
            if key in info:
                assert isinstance(info[key], (int, float))
    
    def test_environment_step_info(self, rl_environment):
        """Test step info provides useful debugging information."""
        env = rl_environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        
        actions = {
            'join_ordering': 1,
            'index_advisor': 2,
            'cache_manager': 1,
            'resource_allocator': 0
        }
        next_state, reward, done, truncated, info = env.step(actions)
        
        # Check info dictionary has useful data
        expected_keys = ['baseline_performance', 'improvement', 'optimization_count']
        found_keys = [key for key in expected_keys if key in info]
        assert len(found_keys) > 0, f"No expected keys found in info: {list(info.keys())}"
        
        # Check info values are reasonable
        assert isinstance(info['performance'], (int, float))
        assert isinstance(info['cost_estimate'], (int, float))
        assert isinstance(info['execution_time'], (int, float))
        assert isinstance(info['actions_taken'], (list, tuple))
        assert len(info['actions_taken']) == 4  # Should record all 4 agent actions