"""
DQN Training Infrastructure - Save/Load Models & Training Loops
"""

import torch
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt

from .dqn_agent import MultiAgentDQN
from .rl_environment import QueryOptimizationEnv
from ..utils.logging import logger

class DQNTrainer:
    """Training infrastructure for DQN agents"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.performance_history = []
        
    def train_episode(self, env: QueryOptimizationEnv, dqn: MultiAgentDQN) -> Dict:
        """Train for one episode"""
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < 50:  # Max steps per episode
            # Get actions from all agents
            actions = dqn.get_actions(state)
            
            # Execute in environment
            next_state, reward, done, truncated, step_info = env.step(actions)
            
            # Store experiences
            dqn.store_experience(state, actions, reward, next_state, done)
            
            # Train agents
            losses = dqn.train_all()
            
            # Update for next step
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
                
        return {
            "reward": total_reward,
            "steps": steps, 
            "losses": losses,
            "final_performance": step_info.get("performance", 0)
        }
    
    def train(self, env: QueryOptimizationEnv, dqn: MultiAgentDQN, 
              num_episodes: int = 1000, save_every: int = 100):
        """Main training loop"""
        
        logger.info(f"Starting DQN training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Train one episode
            episode_result = self.train_episode(env, dqn)
            
            # Record metrics
            self.episode_rewards.append(episode_result["reward"])
            self.performance_history.append(episode_result["final_performance"])
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                stats = dqn.get_stats()
                logger.info(f"Episode {episode}: Reward={episode_result['reward']:.2f}, "
                          f"AvgReward={avg_reward:.2f}, ε={stats['join_ordering']['epsilon']:.3f}")
            
            # Save models periodically
            if episode % save_every == 0 and episode > 0:
                self.save_models(dqn, episode)
                self.save_metrics(episode)
                
        logger.info("Training completed!")
    
    def save_models(self, dqn: MultiAgentDQN, episode: int):
        """Save all agent models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for agent_name, agent in dqn.agents.items():
            model_path = f"{self.model_dir}/{agent_name}_ep{episode}_{timestamp}.pt"
            torch.save({
                'model_state_dict': agent.q_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'step_count': agent.step_count,
                'episode': episode
            }, model_path)
            
        logger.info(f"Models saved at episode {episode}")
    
    def load_models(self, dqn: MultiAgentDQN, episode: int, timestamp: str = None):
        """Load all agent models"""
        
        for agent_name, agent in dqn.agents.items():
            if timestamp:
                model_path = f"{self.model_dir}/{agent_name}_ep{episode}_{timestamp}.pt"
            else:
                # Find latest model for this agent and episode
                files = [f for f in os.listdir(self.model_dir) 
                        if f.startswith(f"{agent_name}_ep{episode}")]
                if not files:
                    logger.warning(f"No saved model found for {agent_name} at episode {episode}")
                    continue
                model_path = f"{self.model_dir}/{sorted(files)[-1]}"
            
            try:
                checkpoint = torch.load(model_path)
                agent.q_network.load_state_dict(checkpoint['model_state_dict'])
                agent.target_network.load_state_dict(checkpoint['target_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                agent.epsilon = checkpoint['epsilon']
                agent.step_count = checkpoint['step_count']
                
                logger.info(f"Loaded {agent_name} from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load {agent_name}: {e}")
    
    def save_metrics(self, episode: int):
        """Save training metrics"""
        metrics = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'performance_history': self.performance_history,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_path = f"{self.model_dir}/metrics_ep{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_progress(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot performance
        ax2.plot(self.performance_history)
        ax2.set_title('Query Performance')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Execution Time (s)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/training_progress.png", dpi=300)
        plt.show()