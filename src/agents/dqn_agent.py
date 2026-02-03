"""
DQN Agent implementation for RL-based query optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.logging import logger


class DQNNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, input_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity: int=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor(np.array([e[4] for e in batch], dtype=np.float32))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    """Single DQN agent for one optimization aspect"""
    
    def __init__(self, name: str, state_size: int, action_size:int):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

        # Network
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0

        # Initialize target network
        self.update_target_network()

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute losss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

        return loss.item()
    
    def update_target_network(self):
        """Copy weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

class MultiAgentDQN:
    """Coordinates mutliple DQN Agents"""

    def __init__(self):
        self.agents = {
            "join_ordering": DQNAgent("join_ordering", 12, 6),
            "index_advisor": DQNAgent("index_advisor", 12, 4), 
            "cache_manager": DQNAgent("cache_manager", 12, 3),
            "resource_allocator": DQNAgent("resource_allocator", 12, 3)
        }
        
        self.episode_count = 0
        
        logger.info("Multi-Agent DQN system ready")

    def get_actions(self, state, deterministic=False):
        """Get actions from all agents for given state as dictionary."""
        agent_actions = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'act'):
                if deterministic and hasattr(agent, 'predict'):
                    agent_actions[agent_name] = agent.predict(state)
                else:
                    agent_actions[agent_name] = agent.act(state)
            else:
                agent_actions[agent_name] = 0  # Default action
        return agent_actions
    
    def get_actions_list(self, state, deterministic=False):
        """Get actions from all agents for given state as list (for integration tests)."""
        agent_actions = self.get_actions(state, deterministic)
        
        # Return as list for testing compatibility
        action_order = ['join_ordering', 'index_advisor', 'cache_manager', 'resource_allocator']
        return [int(agent_actions.get(name, 0)) for name in action_order]

    def store_experience(self, state, actions, reward, next_state, done):
        """Store experience for all agents"""
        for name, agent in self.agents.items():
            if name in actions:
                agent.remember(state, actions[name], reward, next_state, done)  

    def train_all(self):
        """Train all agents"""
        losses = {}
        for name, agent in self.agents.items():
            loss = agent.learn()
            losses[name] = loss
        return losses
    
    def get_stats(self):
        """Get training stats"""
        return {
            name:{
                "epsilon": agent.epsilon,
                "memory_size": len(agent.memory),
                "step_count": agent.step_count
            }
            for name, agent in self.agents.items()
        }
    
    def save_models(self, save_dir: str):
        """Save all agent models"""
        os.makedirs(save_dir, exist_ok=True)
        for agent_name, agent in self.agents.items():
            model_path = os.path.join(save_dir, f"{agent_name}_model.pt")
            torch.save({
                'model_state_dict': agent.q_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'step_count': agent.step_count
            }, model_path)
    
    def load_models(self, save_dir: str):
        """Load all agent models"""
        for agent_name, agent in self.agents.items():
            model_path = os.path.join(save_dir, f"{agent_name}_model.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                agent.q_network.load_state_dict(checkpoint['model_state_dict'])
                agent.target_network.load_state_dict(checkpoint['target_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                agent.epsilon = checkpoint['epsilon']
                agent.step_count = checkpoint['step_count']