import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(rewards, title='Training Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_pnl_curve(pnl, title='PnL Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(pnl, label='Cumulative PnL')
    plt.xlabel('Step')
    plt.ylabel('PnL')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_sharpe_ratio(sharpes, title='Sharpe Ratio Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(sharpes, label='Sharpe Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show() 