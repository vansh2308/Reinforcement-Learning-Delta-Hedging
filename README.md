# Advanced RL Delta Hedging in Quantitative Finance

This project implements advanced reinforcement learning algorithms (DQN, PPO, Actor-Critic) for delta hedging in quantitative finance with real market data integration, comprehensive backtesting, and advanced risk analytics.

## 🚀 Features

### **Core RL Algorithms**
- **DQN (Deep Q-Network)**: Discrete action space with replay buffer
- **PPO (Proximal Policy Optimization)**: Continuous action space with GAE
- **Actor-Critic**: Policy gradient with value function baseline

### **Advanced Pricing Models**
- **Black-Scholes**: Standard geometric Brownian motion
- **Heston Model**: Stochastic volatility with mean reversion
- **Merton Jump Diffusion**: Discontinuous price movements

### **Real Market Data Integration**
- **Yahoo Finance API**: Real stock and option data
- **Historical backtesting**: Use actual market conditions
- **Option chain data**: Real option prices and implied volatility
- **Risk-free rate**: Treasury yield curve integration

### **Advanced Transaction Costs**
- **Linear**: Proportional to trade size
- **Nonlinear**: Increasing cost with larger trades
- **Spread-based**: Bid-ask spread simulation

### **Comprehensive Risk Metrics**
- **VaR/CVaR**: Value at Risk and Conditional VaR
- **Drawdown Analysis**: Maximum and rolling drawdown
- **Sharpe/Sortino Ratios**: Risk-adjusted returns
- **Volatility**: Annualized volatility measures
- **Trading Statistics**: Win rate, trade frequency, etc.

### **Backtesting Framework**
- **Single Agent Backtest**: Detailed performance analysis
- **Multi-Agent Comparison**: Head-to-head strategy comparison
- **Comprehensive Reports**: Automated performance reports
- **Visualization**: 9-panel detailed analysis plots

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Usage Examples

### **Training with Synthetic Data**
```bash
# Train DQN with Heston model and nonlinear costs
python main.py --agent dqn --pricing-model heston --transaction-cost-model nonlinear --episodes 300

# Train PPO with Merton jump diffusion
python main.py --agent ppo --pricing-model merton --episodes 200 --save-model models/ppo_merton.pth
```

### **Real Market Data Training**
```bash
# Train with real AAPL data from 2023
python main.py --agent actor_critic --use-real-data --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31

# Train with specific option parameters
python main.py --agent dqn --use-real-data --symbol TSLA --option-strike 200 --option-expiry 2024-01-19
```

### **Backtesting**
```bash
# Single agent backtest
python main.py --agent ppo --backtest --use-real-data --symbol AAPL --load-model models/ppo_trained.pth

# Multi-agent backtest comparison
python main.py --multi-backtest --use-real-data --symbol MSFT --episodes 5 --save-plots results/comparison.png
```

### **Agent Comparison**
```bash
# Compare all agents with real data
python main.py --compare --use-real-data --symbol GOOGL --episodes 100

# Compare with specific transaction costs
python main.py --compare --transaction-cost-model spread --pricing-model heston
```

### **Evaluation Only**
```bash
# Evaluate trained model without training
python main.py --agent dqn --evaluate-only --load-model models/dqn_trained.pth --use-real-data --symbol AAPL
```

## 📊 Output and Analysis

### **Training Outputs**
- Training curves and convergence plots
- Sharpe ratio evolution
- PnL curves for each episode
- Risk metrics summary (VaR, CVaR, drawdown)

### **Backtest Reports**
- Comprehensive 9-panel analysis
- Market data visualization
- Trading statistics
- Performance comparison tables

### **Risk Analysis**
- Returns distribution with VaR/CVaR lines
- Drawdown analysis with maximum drawdown
- Rolling Sharpe ratio
- Action distribution analysis

## 🏗️ Project Structure

```
rl_delta_hedging/
│
├── envs/
│   ├── delta_hedging_env.py          # Synthetic data environment
│   └── real_data_env.py              # Real market data environment
│
├── agents/
│   ├── dqn_agent.py                  # DQN implementation
│   ├── ppo_agent.py                  # PPO implementation
│   └── actor_critic_agent.py         # Actor-Critic implementation
│
├── utils/
│   ├── data_loader.py                # Real data fetching and processing
│   ├── backtest.py                   # Backtesting framework
│   ├── metrics.py                    # Risk metrics calculations
│   └── plot_utils.py                 # Visualization utilities
│
├── main.py                           # Main training and evaluation script
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🔧 Advanced Configuration

### **Environment Parameters**
- `--pricing-model`: black_scholes, heston, merton
- `--transaction-cost-model`: linear, nonlinear, spread
- `--use-real-data`: Enable real market data
- `--symbol`: Stock symbol (AAPL, TSLA, etc.)
- `--start-date/--end-date`: Date range for real data

### **Training Parameters**
- `--episodes`: Number of training episodes
- `--save-model/--load-model`: Model persistence
- `--evaluate-only`: Evaluation without training

### **Backtesting Parameters**
- `--backtest`: Single agent backtest
- `--multi-backtest`: Multi-agent comparison
- `--save-plots`: Save analysis plots

## 📈 Performance Metrics

The framework calculates and visualizes:
- **Returns**: Total, mean, and standard deviation
- **Risk Metrics**: VaR, CVaR, volatility, drawdown
- **Ratios**: Sharpe, Sortino, Calmar
- **Trading Stats**: Win rate, trade frequency, average trade size

## 📚 References
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Heston Model](https://en.wikipedia.org/wiki/Heston_model)




### Author
- Github - [vansh2308](https://github.com/vansh2308)
- Website - [Vansh Agarwal](https://portfolio-website-self-xi.vercel.app/)