import pygame
import numpy as np
from snake_env import SnakeGameEnv
from model import Agent
import matplotlib.pyplot as plt

def plot(scores1, scores2, mean_scores1, mean_scores2):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Define consistent colors for plotting
    SNAKE1_COLOR = 'blue'
    SNAKE1_MEAN_COLOR = 'lightblue'
    SNAKE2_COLOR = 'darkred'
    SNAKE2_MEAN_COLOR = 'lightcoral'
    
    plt.plot(scores1, color=SNAKE1_COLOR, label='Bluessy Score')
    plt.plot(scores2, color=SNAKE2_COLOR, label='Redish Score')
    plt.plot(mean_scores1, color=SNAKE1_MEAN_COLOR, label='Bluessy Mean')
    plt.plot(mean_scores2, color=SNAKE2_MEAN_COLOR, label='Redish Mean')
    
    plt.ylim(bottom=0)
    plt.legend()
    if scores1:
        plt.text(len(scores1)-1, scores1[-1], str(scores1[-1]))
    if scores2:
        plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.draw()
    plt.pause(0.1)

def train():
    plt.ion()  # interactive mode on
    plt.figure(figsize=(10, 5))
    
    plot_scores1 = []
    plot_scores2 = []
    plot_mean_scores1 = []
    plot_mean_scores2 = []
    total_score1 = 0
    total_score2 = 0
    record = 0
    
    # Initialize environment and agents
    env = SnakeGameEnv()
    # Allow RL training to run uninterrupted (mouse events won’t block training)
    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
    pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
    
    agent1 = Agent()
    agent2 = Agent()
    
    try:
        game_number = 0
        while True:
            game_number += 1
            
            # Reset environment and get initial states
            state1, state2 = env.reset()
            done = False
            while not done:
                # Obtain actions from agents
                action1 = agent1.get_action(state1)
                action2 = agent2.get_action(state2)
                
                # Step the environment and train step by step
                (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
                agent1.train_step([state1], [action1], [reward1], [new_state1], [done])
                agent2.train_step([state2], [action2], [reward2], [new_state2], [done])
                
                state1 = new_state1
                state2 = new_state2
                
                # Render the game window (non-blocking)
                env.render(game_number)
                
                if done:
                    break
            
            # Train long memory after each game
            agent1.train_long_memory()
            agent2.train_long_memory()
            agent1.n_games += 1
            agent2.n_games += 1
            
            # Update scores for plotting
            plot_scores1.append(env.score1)
            plot_scores2.append(env.score2)
            total_score1 += env.score1
            total_score2 += env.score2
            mean_score1 = total_score1 / agent1.n_games
            mean_score2 = total_score2 / agent2.n_games
            plot_mean_scores1.append(mean_score1)
            plot_mean_scores2.append(mean_score2)
            
            if env.score1 > record or env.score2 > record:
                record = max(env.score1, env.score2)
            
            print(f'Game {agent1.n_games} | Bluessy: {env.score1} | Redish: {env.score2} | Record: {record}')
            plot(plot_scores1, plot_scores2, plot_mean_scores1, plot_mean_scores2)
            
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    finally:
        env.close()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    train()
