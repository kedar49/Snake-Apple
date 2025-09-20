import pygame
import numpy as np
from snake_env import SnakeGameEnv
from model import Agent
import matplotlib.pyplot as plt
from logger import TrainingLogger
import os
import time

def plot(scores1, scores2, mean_scores1, mean_scores2):
    try:
        pygame.event.set_blocked(None)
        
        if not plt.get_fignums():
            plt.figure(figsize=(6, 4), dpi=80)
        else:
            plt.clf()
            plt.figure(plt.get_fignums()[0])
        
        plt.get_current_fig_manager().window.lift()
        
        plt.title('Training Progress', pad=10, fontsize=12, color='black')
        plt.xlabel('Number of Games', fontsize=10, color='black')
        plt.ylabel('Score', fontsize=10, color='black')
        plt.gca().set_facecolor('#f0f0f0')
        plt.gcf().set_facecolor('#ffffff')
        
        SNAKE1_COLOR = '#0066cc'
        SNAKE1_MEAN_COLOR = '#66b3ff'
        SNAKE2_COLOR = '#cc0000'
        SNAKE2_MEAN_COLOR = '#ff6666'
        
        plt.plot(scores1, color=SNAKE1_COLOR, label='Bluessy Score', alpha=0.5, linewidth=1)
        plt.plot(scores2, color=SNAKE2_COLOR, label='Redish Score', alpha=0.5, linewidth=1)
        plt.plot(mean_scores1, color=SNAKE1_MEAN_COLOR, label='Bluessy Mean', linewidth=2)
        plt.plot(mean_scores2, color=SNAKE2_MEAN_COLOR, label='Redish Mean', linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        plt.legend(loc='upper left', fontsize=8, framealpha=0.7)
        
        if scores1:
            plt.text(len(scores1)-1, scores1[-1], f'{scores1[-1]}', color=SNAKE1_COLOR, fontsize=8)
        if scores2:
            plt.text(len(scores2)-1, scores2[-1], f'{scores2[-1]}', color=SNAKE2_COLOR, fontsize=8)
        
        plt.tight_layout()
        
        try:
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)
        except Exception as draw_error:
            print(f"Plot update error: {draw_error}")
            plt.figure()
            
    except Exception as e:
        print(f"Plotting error: {e}")
        plt.close('all')
        plt.figure(figsize=(6, 4), dpi=80)
    finally:
        pygame.event.set_allowed(None)
                
def train():
    try:
        logger = TrainingLogger()
        logger.logger.info("Starting enhanced Snake RL training with Double DQN, Dueling DQN, and Prioritized Replay")
        
        plt.ion()
        plt.figure(figsize=(12, 8), dpi=100)
        
        env = SnakeGameEnv()
        pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
        pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
        
        agent1 = Agent(input_size=25, use_dueling=True, use_double_dqn=True, use_prioritized_replay=True)
        agent2 = Agent(input_size=25, use_dueling=True, use_double_dqn=True, use_prioritized_replay=True)
        
        game_number = 0
        best_score = 0
        save_frequency = 50
        
        logger.logger.info(f"Using device: {agent1.model.device if hasattr(agent1.model, 'device') else 'CPU'}")
        
        while True:
            game_number += 1
            state1, state2 = env.reset()
            done = False
            step_count = 0
            episode_rewards1 = []
            episode_rewards2 = []
            
            while not done:
                for event in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        logger.logger.info('Exiting game...')
                        env.close()
                        plt.ioff()
                        plt.close('all')
                        return
                
                action1 = agent1.get_action(state1, train=True)
                action2 = agent2.get_action(state2, train=True)
                
                action1_idx = np.argmax(action1)
                action2_idx = np.argmax(action2)
                
                (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
                
                agent1.remember(state1, action1_idx, reward1, new_state1, done)
                agent2.remember(state2, action2_idx, reward2, new_state2, done)
                
                agent1.train_long_memory()
                agent2.train_long_memory()
                
                logger.log_step(reward1, reward2)
                episode_rewards1.append(reward1)
                episode_rewards2.append(reward2)
                
                state1, state2 = new_state1, new_state2
                
                if env.paused:
                    while env.paused:
                        env.render(game_number)
                        time.sleep(0.1)
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                env.paused = False
                                break
                
                env.render(game_number)
                
                if env.training_speed != 1.0:
                    time.sleep((1.0 / env.training_speed) * 0.1)
                
                step_count += 1
                if step_count % 20 == 0:
                    print(f'\rGame {game_number} | Step {step_count} | Bluessy: {env.score1} | Redish: {env.score2} | Epsilon: {agent1.epsilon:.3f} | Speed: {env.training_speed:.1f}x', end='')
            
            agent1.n_games += 1
            agent2.n_games += 1
            
            winner = None
            if env.score1 > env.score2:
                winner = 1
            elif env.score2 > env.score1:
                winner = 2
            
            logger.log_game(game_number, env.score1, env.score2, 
                          agent1.get_metrics(), agent2.get_metrics(), 
                          step_count, winner)
            
            current_best = max(env.score1, env.score2)
            if current_best > best_score:
                best_score = current_best
                logger.logger.info(f"New best score: {best_score}")
            
            if game_number % save_frequency == 0:
                agent1.save_model(f"agent1_game_{game_number}.pth")
                agent2.save_model(f"agent2_game_{game_number}.pth")
                logger.save_metrics(f"metrics_game_{game_number}.json")
                
                plot_path = os.path.join(logger.log_dir, f"training_plot_game_{game_number}.png")
                logger.plot_training_progress(plot_path)
            
            if game_number % 10 == 0:
                try:
                    logger.plot_training_progress()
                    plt.pause(0.1)
                except Exception as e:
                    logger.logger.warning(f"Plot update error: {e}")
            
            if game_number % 100 == 0:
                stats = logger.get_summary_stats()
                logger.logger.info(f"Summary after {game_number} games:")
                logger.logger.info(f"  Average scores: Bluessy={stats['avg_score1']:.2f}, Redish={stats['avg_score2']:.2f}")
                logger.logger.info(f"  Best scores: Bluessy={stats['max_score1']}, Redish={stats['max_score2']}")
                logger.logger.info(f"  Win rate: {stats['win_rate1']:.2%}")
                logger.logger.info(f"  Average game length: {stats['avg_game_length']:.1f} steps")
            
    except KeyboardInterrupt:
        logger.logger.info('Training interrupted by user')
    except Exception as e:
        logger.logger.error(f"Training error: {e}")
        raise
    finally:
        try:
            agent1.save_model("agent1_final.pth")
            agent2.save_model("agent2_final.pth")
            logger.save_metrics("metrics_final.json")
            logger.plot_training_progress(os.path.join(logger.log_dir, "final_training_plot.png"))
            
            env.close()
            plt.ioff()
            plt.close('all')
            
            stats = logger.get_summary_stats()
            logger.logger.info("Final Training Summary:")
            logger.logger.info(f"  Total games: {stats['total_games']}")
            logger.logger.info(f"  Final average scores: Bluessy={stats['avg_score1']:.2f}, Redish={stats['avg_score2']:.2f}")
            logger.logger.info(f"  Best scores: Bluessy={stats['max_score1']}, Redish={stats['max_score2']}")
            logger.logger.info(f"  Final win rate: {stats['win_rate1']:.2%}")
            
        except Exception as e:
            logger.logger.error(f"Error during cleanup: {e}")

if __name__ == '__main__':
    train()
