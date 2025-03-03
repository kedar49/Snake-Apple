import pygame
import numpy as np
from snake_env import SnakeGameEnv
from model import Agent
import matplotlib.pyplot as plt

def plot(scores1, scores2, mean_scores1, mean_scores2):
    try:
        pygame.event.set_blocked(None)
        
        if not plt.get_fignums():
            plt.figure(figsize=(6, 4), dpi=80)
        else:
            plt.clf()
            plt.figure(plt.get_fignums()[0])
        
        plt.get_current_fig_manager().window.lift()
        
        # Set plot style
        plt.title('Training Progress', pad=10, fontsize=12, color='black')
        plt.xlabel('Number of Games', fontsize=10, color='black')
        plt.ylabel('Score', fontsize=10, color='black')
        plt.gca().set_facecolor('#f0f0f0')
        plt.gcf().set_facecolor('#ffffff')
        
        # Snake colors
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
        plt.ion()
        plt.figure(figsize=(6, 4), dpi=80)
        
        plot_scores1, plot_scores2 = [], []
        plot_mean_scores1, plot_mean_scores2 = [], []
        total_score1, total_score2 = 0, 0
        record = 0
        
        env = SnakeGameEnv()
        pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
        pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
        
        agent1, agent2 = Agent(), Agent()
        game_number = 0
        
        while True:
            game_number += 1
            state1, state2 = env.reset()
            done = False
            step_count = 0
            
            while not done:
                # Handle exit events
                for event in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print('\nExiting game...')
                        env.close()
                        plt.ioff()
                        plt.close('all')
                        return
                
                action1 = agent1.get_action(state1)
                action2 = agent2.get_action(state2)
                
                (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
                
                agent1.train_step([state1], [action1], [reward1], [new_state1], [done])
                agent2.train_step([state2], [action2], [reward2], [new_state2], [done])
                
                state1, state2 = new_state1, new_state2
                env.render(game_number)
                
                step_count += 1
                if step_count % 10 == 0:
                    print(f'\rGame {game_number} | Step {step_count} | Bluessy: {env.score1} | Redish: {env.score2} | Record: {record}', end='')
            
            print(f'\nGame {game_number} Complete | Bluessy: {env.score1} | Redish: {env.score2} | Record: {record}')
            plot(plot_scores1, plot_scores2, plot_mean_scores1, plot_mean_scores2)
            
            try:
                manager = plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    manager.window.lift()
                    manager.window.attributes('-topmost', True)
                    manager.window.attributes('-topmost', False)
                plt.pause(0.1)
            except Exception as e:
                print(f"Window management error: {e}")
                plt.close('all')
                plt.figure(figsize=(6, 4), dpi=80)
            
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        try:
            env.close()
            plt.ioff()
            plt.close('all')
        except:
            pass

if __name__ == '__main__':
    train()
