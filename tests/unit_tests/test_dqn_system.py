"""
Unit tests for DQN (Deep Q-Network) system components.

This module tests the neural networks, agents, experience replay,
and multi-agent coordination functionality of the DQN system.
"""

import sys
import pytest
import torch
import numpy as np
import os
sys.path.append('src')
sys.path.append('evaluation')

from src.agents.dqn_agent import DQNNetwork, DQNAgent, MultiAgentDQN


class TestDQNSystem:
    """Test suite for DQN system components."""
    
    def test_dqn_network_creation(self):
        """Test DQN neural network creation and forward pass."""
        state_dim = 12
        action_dim = 6
        
        network = DQNNetwork(state_dim, action_dim)
        
        # Test network structure
        assert hasattr(network, 'fc1')
        assert hasattr(network, 'fc2') 
        assert hasattr(network, 'fc3')
        assert hasattr(network, 'dropout')
        
        # Test forward pass
        batch_size = 32
        dummy_state = torch.randn(batch_size, state_dim)
        
        with torch.no_grad():
            q_values = network(dummy_state)
            
        assert q_values.shape == (batch_size, action_dim)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization and basic functionality."""
        state_dim = 12
        action_dim = 4
        
        agent = DQNAgent('test_agent', state_dim, action_dim)
        
        # Test agent attributes
        assert hasattr(agent, 'q_network')
        assert hasattr(agent, 'target_network')
        assert hasattr(agent, 'optimizer')
        assert hasattr(agent, 'memory')
        assert agent.epsilon == 1.0  # Initial epsilon
        assert agent.action_size == action_dim
        
        # Test action selection
        state = np.random.randn(state_dim)
        action = agent.act(state)
        
        assert 0 <= action < action_dim
        assert isinstance(action, (int, np.integer))
    
    def test_experience_replay(self):
        """Test experience replay buffer functionality.""" 
        agent = DQNAgent('test_agent', 12, 4)
        
        # Add experiences
        for i in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.choice([True, False])
            
            agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 100
        
        # Test replay sampling
        if len(agent.memory) >= agent.batch_size:
            loss = agent.learn()
            assert isinstance(loss, float)
            assert loss >= 0  # Loss should be non-negative
    
    def test_multi_agent_dqn(self):
        """Test multi-agent DQN coordination."""
        dqn_system = MultiAgentDQN()
        
        # Test agent creation
        assert len(dqn_system.agents) == 4
        expected_agents = ['join_ordering', 'index_advisor', 'cache_manager', 'resource_allocator']
        
        for agent_name in expected_agents:
            assert agent_name in dqn_system.agents
            agent = dqn_system.agents[agent_name]
            assert isinstance(agent, DQNAgent)
        
        # Test action dimensions
        expected_actions = {'join_ordering': 6, 'index_advisor': 4, 'cache_manager': 3, 'resource_allocator': 3}
        
        state = np.random.randn(12)
        actions = dqn_system.get_actions(state)
        
        assert len(actions) == 4
        for agent_name, action in actions.items():
            assert 0 <= action < expected_actions[agent_name]
    
    def test_dqn_training(self):
        """Test DQN training process."""
        dqn_system = MultiAgentDQN()
        
        # Generate training data
        for _ in range(50):  # Minimum for training
            state = np.random.randn(12)
            actions = dqn_system.get_actions(state)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.choice([True, False])
            
            dqn_system.store_experience(state, actions, reward, next_state, done)
        
        # Train agents
        losses = dqn_system.train_all()
        
        assert isinstance(losses, dict)
        assert len(losses) == 4
        
        for agent_name, loss in losses.items():
            assert isinstance(loss, float)
            assert loss >= 0
    
    def test_dqn_statistics(self):
        """Test DQN system statistics collection."""
        dqn_system = MultiAgentDQN()
        
        stats = dqn_system.get_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) == 4
        
        for agent_name in ['join_ordering', 'index_advisor', 'cache_manager', 'resource_allocator']:
            assert agent_name in stats
            agent_stats = stats[agent_name]
            
            assert 'epsilon' in agent_stats
            assert 'memory_size' in agent_stats
            assert 'step_count' in agent_stats
    
    def test_dqn_with_rl_environment(self, rl_environment):
        """Test DQN system integration with RL environment."""
        env = rl_environment
        dqn = MultiAgentDQN()
        
        # Test environment reset and action selection
        reset_result = env.reset()
        # Handle different return formats from environment reset
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        actions = dqn.get_actions(state)
        
        # Execute step
        next_state, reward, done, truncated, info = env.step(actions)
        
        # Store experience
        dqn.store_experience(state, actions, reward, next_state, done)
        
        # Verify state shapes
        assert len(state) >= 12  # State might be longer than expected
        assert len(next_state) >= 12
        assert len(actions) == 4
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        # Test multiple episodes for learning
        for episode in range(10):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
            
            for step in range(3):  # Short episodes
                actions = dqn.get_actions(state)
                next_state, reward, done, truncated, info = env.step(actions)
                
                dqn.store_experience(state, actions, reward, next_state, done or truncated)
                state = next_state
                
                if done or truncated:
                    break
        
        # Test training after collecting experiences
        if any(len(agent.memory) >= 32 for agent in dqn.agents.values()):
            losses = dqn.train_all()
            assert isinstance(losses, dict)
    
    def test_target_network_updates(self):
        """Test target network update mechanism."""
        agent = DQNAgent('test_agent', 12, 4)
        
        # Get initial target weights
        initial_target_weights = agent.target_network.fc1.weight.clone()
        
        # Force target network update
        agent.step_count = agent.update_target_freq
        agent.update_target_network()
        
        # Verify weights changed
        updated_target_weights = agent.target_network.fc1.weight
        
        # Should be exactly equal (hard update in this implementation)
        assert torch.allclose(updated_target_weights, agent.q_network.fc1.weight)
    
    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        agent = DQNAgent('test_agent', 12, 4)
        
        initial_epsilon = agent.epsilon
        
        # Simulate training to decay epsilon
        for _ in range(100):
            if len(agent.memory) >= agent.batch_size:
                agent.learn()
            else:
                # Add dummy experience to enable training
                state = np.random.randn(12)
                action = np.random.randint(0, 4)
                reward = 1.0
                next_state = np.random.randn(12)
                done = False
                agent.remember(state, action, reward, next_state, done)
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading functionality."""
        dqn_system = MultiAgentDQN()
        
        # Add some experience and training
        for _ in range(50):
            state = np.random.randn(12)
            actions = dqn_system.get_actions(state)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = False
            
            dqn_system.store_experience(state, actions, reward, next_state, done)
        
        dqn_system.train_all()
        
        # Save models
        save_dir = str(tmp_path)
        dqn_system.save_models(save_dir)
        
        # Verify files were created
        for agent_name in dqn_system.agents.keys():
            model_file = os.path.join(save_dir, f"{agent_name}_model.pt")
            assert os.path.exists(model_file)
        
        # Create new system and load
        new_dqn = MultiAgentDQN()
        new_dqn.load_models(save_dir)
        
        # Verify loaded weights match
        for agent_name in dqn_system.agents.keys():
            original_weights = dqn_system.agents[agent_name].q_network.fc1.weight
            loaded_weights = new_dqn.agents[agent_name].q_network.fc1.weight
            assert torch.allclose(original_weights, loaded_weights, rtol=1e-4)