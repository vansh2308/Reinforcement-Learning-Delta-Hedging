# RL Delta Hedging in Quantitative Finance

This project implements advanced reinforcement learning algorithms (DQN, PPO, Actor-Critic) for delta hedging in quantitative finance. The codebase is modular, supports custom environments, and provides detailed performance analysis including PnL and Sharpe ratio plots.

## Features
- Modular RL agents: DQN, PPO, Actor-Critic (PyTorch)
- Custom OpenAI Gymnasium environment for delta hedging
- Training and evaluation scripts
- Plots: training curves, PnL, Sharpe ratio
- Metrics: Sharpe ratio, cumulative PnL, etc.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run training and evaluation:
```bash
python main.py --agent dqn
python main.py --agent ppo
python main.py --agent actor_critic
```

## Directory Structure
- `envs/`: Custom environments
- `agents/`: RL agent implementations
- `utils/`: Plotting and metrics utilities
- `main.py`: Training and evaluation entry point

## References
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI Gymnasium](https://gymnasium.farama.org/) 