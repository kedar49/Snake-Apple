#!/usr/bin/env python3
"""
Test script for the enhanced Snake RL environment
"""
import sys
import os
import time

def test_imports():
    """Test that all modules can be imported"""
    try:
        import torch
        import pygame
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ Basic imports successful")
        
        from snake_env import SnakeGameEnv
        from model import Agent
        from logger import TrainingLogger
        from config import get_config_summary
        print("✓ Project imports successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_environment():
    """Test the game environment"""
    try:
        from snake_env import SnakeGameEnv
        
        print("Testing environment initialization...")
        env = SnakeGameEnv()
        print("✓ Environment initialized")
        
        print("Testing environment reset...")
        state1, state2 = env.reset()
        print(f"✓ Environment reset - State sizes: {len(state1)}, {len(state2)}")
        
        print("Testing environment step...")
        action1 = [1, 0, 0]  # Straight
        action2 = [0, 1, 0]  # Right
        (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
        print(f"✓ Environment step - Rewards: {reward1:.2f}, {reward2:.2f}, Done: {done}")
        
        print("Testing environment render...")
        env.render(1)
        print("✓ Environment render successful")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent():
    """Test the enhanced agent"""
    try:
        from model import Agent
        
        print("Testing agent initialization...")
        agent = Agent(input_size=25, use_dueling=True, use_double_dqn=True, use_prioritized_replay=True)
        print("✓ Agent initialized")
        
        print("Testing agent action...")
        state = [0] * 25  # Dummy state
        action = agent.get_action(state, train=True)
        print(f"✓ Agent action: {action}")
        
        print("Testing agent memory...")
        agent.remember(state, 0, 1.0, state, False)
        print("✓ Agent memory working")
        
        print("Testing agent training...")
        agent.train_long_memory()
        print("✓ Agent training working")
        
        print("Testing agent metrics...")
        metrics = agent.get_metrics()
        print(f"✓ Agent metrics: {metrics}")
        
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_logger():
    """Test the logging system"""
    try:
        from logger import TrainingLogger
        
        print("Testing logger initialization...")
        logger = TrainingLogger()
        print("✓ Logger initialized")
        
        print("Testing logger game logging...")
        logger.log_game(1, 5, 3, {'avg_loss': 0.1, 'avg_q_value': 2.5, 'epsilon': 0.8}, 
                       {'avg_loss': 0.2, 'avg_q_value': 2.3, 'epsilon': 0.8}, 100, 1)
        print("✓ Logger game logging working")
        
        print("Testing logger step logging...")
        logger.log_step(0.1, 0.2)
        print("✓ Logger step logging working")
        
        print("Testing logger metrics save...")
        logger.save_metrics("test_metrics.json")
        print("✓ Logger metrics save working")
        
        return True
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False

def test_config():
    """Test the configuration system"""
    try:
        from config import get_config_summary, get_device, create_directories
        
        print("Testing configuration...")
        config = get_config_summary()
        print(f"✓ Configuration loaded - Device: {config['device']}")
        
        print("Testing directory creation...")
        create_directories()
        print("✓ Directories created")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def run_quick_training_test():
    """Run a quick training test"""
    try:
        from snake_env import SnakeGameEnv
        from model import Agent
        from logger import TrainingLogger
        
        print("Running quick training test...")
        
        # Initialize components
        logger = TrainingLogger()
        env = SnakeGameEnv()
        agent1 = Agent(input_size=25, use_dueling=True, use_double_dqn=True, use_prioritized_replay=True)
        agent2 = Agent(input_size=25, use_dueling=True, use_double_dqn=True, use_prioritized_replay=True)
        
        # Run a few training steps
        for game in range(3):
            state1, state2 = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 50:  # Limit steps for quick test
                action1 = agent1.get_action(state1, train=True)
                action2 = agent2.get_action(state2, train=True)
                
                action1_idx = action1.index(1)
                action2_idx = action2.index(1)
                
                (new_state1, new_state2), reward1, reward2, done = env.step(action1, action2)
                
                agent1.remember(state1, action1_idx, reward1, new_state1, done)
                agent2.remember(state2, action2_idx, reward2, new_state2, done)
                
                agent1.train_long_memory()
                agent2.train_long_memory()
                
                logger.log_step(reward1, reward2)
                
                state1, state2 = new_state1, new_state2
                env.render(game + 1)
                
                step_count += 1
            
            # Log game results
            winner = 1 if env.score1 > env.score2 else 2 if env.score2 > env.score1 else None
            logger.log_game(game + 1, env.score1, env.score2, 
                          agent1.get_metrics(), agent2.get_metrics(), 
                          step_count, winner)
            
            agent1.n_games += 1
            agent2.n_games += 1
        
        env.close()
        print("✓ Quick training test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED SNAKE RL - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Agent Test", test_agent),
        ("Logger Test", test_logger),
        ("Config Test", test_config),
        ("Quick Training Test", run_quick_training_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The enhanced Snake RL system is ready to use.")
        print("\nTo start training, run: python train.py")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  H - Toggle Help")
        print("  +/- - Speed Up/Down")
        print("  R - Reset Speed")
        print("  ESC - Exit")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
