import logging
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Training metrics storage
        self.metrics = {
            'games': [],
            'scores1': [],
            'scores2': [],
            'mean_scores1': [],
            'mean_scores2': [],
            'losses1': [],
            'losses2': [],
            'q_values1': [],
            'q_values2': [],
            'epsilon1': [],
            'epsilon2': [],
            'rewards1': [],
            'rewards2': [],
            'game_lengths': [],
            'win_rates': []
        }
        
        self.game_count = 0
        self.episode_rewards1 = []
        self.episode_rewards2 = []
        
    def log_game(self, game_number, score1, score2, agent1_metrics, agent2_metrics, 
                 game_length, winner=None):
        """Log game results and agent metrics"""
        self.game_count = game_number
        
        # Store scores
        self.metrics['games'].append(game_number)
        self.metrics['scores1'].append(score1)
        self.metrics['scores2'].append(score2)
        
        # Calculate moving averages
        window = min(100, len(self.metrics['scores1']))
        mean_score1 = np.mean(self.metrics['scores1'][-window:])
        mean_score2 = np.mean(self.metrics['scores2'][-window:])
        
        self.metrics['mean_scores1'].append(mean_score1)
        self.metrics['mean_scores2'].append(mean_score2)
        
        # Store agent metrics
        self.metrics['losses1'].append(agent1_metrics.get('avg_loss', 0))
        self.metrics['losses2'].append(agent2_metrics.get('avg_loss', 0))
        self.metrics['q_values1'].append(agent1_metrics.get('avg_q_value', 0))
        self.metrics['q_values2'].append(agent2_metrics.get('avg_q_value', 0))
        self.metrics['epsilon1'].append(agent1_metrics.get('epsilon', 0))
        self.metrics['epsilon2'].append(agent2_metrics.get('epsilon', 0))
        
        # Store episode rewards
        if self.episode_rewards1:
            self.metrics['rewards1'].append(np.sum(self.episode_rewards1))
            self.metrics['rewards2'].append(np.sum(self.episode_rewards2))
            self.episode_rewards1 = []
            self.episode_rewards2 = []
        
        self.metrics['game_lengths'].append(game_length)
        
        # Calculate win rate
        if winner is not None:
            recent_games = self.metrics['games'][-100:]
            recent_winners = [1 if s1 > s2 else 2 if s2 > s1 else 0 
                            for s1, s2 in zip(self.metrics['scores1'][-100:], 
                                            self.metrics['scores2'][-100:])]
            win_rate = recent_winners.count(1) / len([w for w in recent_winners if w != 0]) if any(w != 0 for w in recent_winners) else 0
            self.metrics['win_rates'].append(win_rate)
        
        # Log to console and file
        self.logger.info(f"Game {game_number}: Bluessy={score1}, Redish={score2}, "
                        f"Mean: B={mean_score1:.2f}, R={mean_score2:.2f}, "
                        f"Length={game_length}, Winner={'Bluessy' if winner == 1 else 'Redish' if winner == 2 else 'Tie'}")
    
    def log_step(self, reward1, reward2):
        """Log step rewards"""
        self.episode_rewards1.append(reward1)
        self.episode_rewards2.append(reward2)
    
    def save_metrics(self, filename=None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def plot_training_progress(self, save_path=None):
        """Create comprehensive training plots"""
        if not self.metrics['games']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress Dashboard', fontsize=16)
        
        # Scores plot
        axes[0, 0].plot(self.metrics['games'], self.metrics['scores1'], 
                       alpha=0.6, color='#0066cc', label='Bluessy Score')
        axes[0, 0].plot(self.metrics['games'], self.metrics['scores2'], 
                       alpha=0.6, color='#cc0000', label='Redish Score')
        axes[0, 0].plot(self.metrics['games'], self.metrics['mean_scores1'], 
                       color='#0066cc', linewidth=2, label='Bluessy Mean')
        axes[0, 0].plot(self.metrics['games'], self.metrics['mean_scores2'], 
                       color='#cc0000', linewidth=2, label='Redish Mean')
        axes[0, 0].set_title('Scores Over Time')
        axes[0, 0].set_xlabel('Games')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        if self.metrics['losses1']:
            axes[0, 1].plot(self.metrics['games'], self.metrics['losses1'], 
                           color='#0066cc', label='Bluessy Loss')
            axes[0, 1].plot(self.metrics['games'], self.metrics['losses2'], 
                           color='#cc0000', label='Redish Loss')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Games')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Q-values plot
        if self.metrics['q_values1']:
            axes[0, 2].plot(self.metrics['games'], self.metrics['q_values1'], 
                           color='#0066cc', label='Bluessy Q-Value')
            axes[0, 2].plot(self.metrics['games'], self.metrics['q_values2'], 
                           color='#cc0000', label='Redish Q-Value')
            axes[0, 2].set_title('Average Q-Values')
            axes[0, 2].set_xlabel('Games')
            axes[0, 2].set_ylabel('Q-Value')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Epsilon decay
        if self.metrics['epsilon1']:
            axes[1, 0].plot(self.metrics['games'], self.metrics['epsilon1'], 
                           color='#0066cc', label='Bluessy Epsilon')
            axes[1, 0].plot(self.metrics['games'], self.metrics['epsilon2'], 
                           color='#cc0000', label='Redish Epsilon')
            axes[1, 0].set_title('Exploration Rate (Epsilon)')
            axes[1, 0].set_xlabel('Games')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Game lengths
        axes[1, 1].plot(self.metrics['games'], self.metrics['game_lengths'], 
                       color='green', alpha=0.7)
        axes[1, 1].set_title('Game Lengths')
        axes[1, 1].set_xlabel('Games')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Win rates
        if self.metrics['win_rates']:
            axes[1, 2].plot(self.metrics['games'], self.metrics['win_rates'], 
                           color='purple', linewidth=2)
            axes[1, 2].set_title('Bluessy Win Rate (100 games)')
            axes[1, 2].set_xlabel('Games')
            axes[1, 2].set_ylabel('Win Rate')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plot saved to {save_path}")
        
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if not self.metrics['games']:
            return {}
        
        recent_games = min(100, len(self.metrics['scores1']))
        
        return {
            'total_games': len(self.metrics['games']),
            'avg_score1': np.mean(self.metrics['scores1'][-recent_games:]),
            'avg_score2': np.mean(self.metrics['scores2'][-recent_games:]),
            'max_score1': np.max(self.metrics['scores1']),
            'max_score2': np.max(self.metrics['scores2']),
            'avg_game_length': np.mean(self.metrics['game_lengths'][-recent_games:]),
            'win_rate1': self.metrics['win_rates'][-1] if self.metrics['win_rates'] else 0,
            'current_epsilon1': self.metrics['epsilon1'][-1] if self.metrics['epsilon1'] else 0,
            'current_epsilon2': self.metrics['epsilon2'][-1] if self.metrics['epsilon2'] else 0
        }
