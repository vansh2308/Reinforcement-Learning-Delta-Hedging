import argparse
import numpy as np
from envs.delta_hedging_env import DeltaHedgingEnv
from envs.real_data_env import RealDataDeltaHedgingEnv
import torch
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.actor_critic_agent import ActorCriticAgent
from utils.plot_utils import (plot_training_curve, plot_pnl_curve, plot_sharpe_ratio, 
                             plot_var_cvar, plot_drawdown, plot_risk_metrics, plot_comparison)
from utils.metrics import (compute_sharpe_ratio, compute_cumulative_pnl, compute_var, 
                          compute_cvar, compute_drawdown, compute_volatility, compute_sortino_ratio)
from utils.backtest import BacktestEngine, MultiAgentBacktest

AGENT_MAP = {
    'dqn': DQNAgent,
    'ppo': PPOAgent,
    'actor_critic': ActorCriticAgent
}

def train(agent_name, episodes=200, pricing_model='black_scholes', transaction_cost_model='linear', 
          save_model=None, load_model=None, evaluate_only=False, use_real_data=False,
          symbol='AAPL', start_date='2023-01-01', end_date=None, option_strike=None, option_expiry=None):
    
    # Choose environment
    if use_real_data:
        env = RealDataDeltaHedgingEnv(
            symbol=symbol, start_date=start_date, end_date=end_date,
            option_strike=option_strike, option_expiry=option_expiry,
            transaction_cost_model=transaction_cost_model
        )
    else:
        env = DeltaHedgingEnv(pricing_model=pricing_model, transaction_cost_model=transaction_cost_model)
    
    state_dim = env.observation_space.shape[0]
    action_dim = 5 if agent_name == 'dqn' else env.action_space.shape[0]
    agent = AGENT_MAP[agent_name](state_dim, action_dim)
    
    if load_model:
        print(f"Loading model from {load_model}")
        agent.load(load_model)
    
    episode_rewards = []
    sharpe_ratios = []
    all_returns = []
    all_pnl_series = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        rewards = []
        pnl = []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            pnl.append(env.pnl)
            
            if not evaluate_only:
                if agent_name == 'dqn':
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    agent.train_step()
                else:
                    if agent_name == 'ppo':
                        value = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()
                        agent.store((state, action, reward, done, value))
                    else:
                        agent.store((state, action, reward, next_state, done))
            
            state = next_state
        
        if not evaluate_only:
            if agent_name == 'ppo':
                agent.train_step(next_state)
            elif agent_name == 'actor_critic':
                agent.train_step()
        
        episode_rewards.append(np.sum(rewards))
        sharpe = compute_sharpe_ratio(rewards)
        sharpe_ratios.append(sharpe)
        all_returns.extend(rewards)
        all_pnl_series.append(pnl)
        
        print(f"Episode {ep+1}/{episodes} | Reward: {episode_rewards[-1]:.2f} | Sharpe: {sharpe:.2f}")
    
    # Comprehensive analysis and plotting
    print(f"\n=== {agent_name.upper()} Analysis ===")
    if use_real_data:
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
    else:
        print(f"Pricing Model: {pricing_model}")
    print(f"Transaction Cost Model: {transaction_cost_model}")
    
    # Basic plots
    plot_training_curve(episode_rewards, title=f'{agent_name.upper()} Training Curve')
    plot_sharpe_ratio(sharpe_ratios, title=f'{agent_name.upper()} Sharpe Ratio')
    
    # Risk analysis plots
    if len(all_returns) > 0:
        plot_var_cvar(all_returns, title=f'{agent_name.upper()} VaR/CVaR Analysis')
    
    if len(all_pnl_series) > 0:
        final_pnl = all_pnl_series[-1]
        plot_pnl_curve(final_pnl, title=f'{agent_name.upper()} PnL Curve (Last Episode)')
        plot_drawdown(final_pnl, title=f'{agent_name.upper()} Drawdown Analysis')
        
        # Comprehensive risk metrics
        plot_risk_metrics(all_returns, final_pnl, title=f'{agent_name.upper()} Risk Metrics Summary')
    
    if save_model and not evaluate_only:
        print(f"Saving model to {save_model}")
        agent.save(save_model)
    
    return episode_rewards, sharpe_ratios, all_pnl_series[-1] if all_pnl_series else []

def backtest(agent_name, use_real_data=False, symbol='AAPL', start_date='2023-01-01', 
            end_date=None, option_strike=None, option_expiry=None, transaction_cost_model='linear',
            load_model=None, episodes=1, save_plots=None):
    """Run backtest with comprehensive analysis"""
    
    # Choose environment
    if use_real_data:
        env = RealDataDeltaHedgingEnv(
            symbol=symbol, start_date=start_date, end_date=end_date,
            option_strike=option_strike, option_expiry=option_expiry,
            transaction_cost_model=transaction_cost_model
        )
    else:
        env = DeltaHedgingEnv(transaction_cost_model=transaction_cost_model)
    
    state_dim = env.observation_space.shape[0]
    action_dim = 5 if agent_name == 'dqn' else env.action_space.shape[0]
    agent = AGENT_MAP[agent_name](state_dim, action_dim)
    
    if load_model:
        print(f"Loading model from {load_model}")
        agent.load(load_model)
    
    # Run backtest
    backtest_engine = BacktestEngine(env, agent)
    results = backtest_engine.run_backtest(episodes)
    
    # Generate comprehensive analysis
    backtest_engine.plot_backtest_results(save_path=save_plots)
    backtest_engine.generate_report()
    
    return results

def compare_agents(agents, episodes=200, pricing_model='black_scholes', transaction_cost_model='linear',
                  use_real_data=False, symbol='AAPL', start_date='2023-01-01', end_date=None):
    """Compare multiple agents"""
    results = {}
    
    for agent_name in agents:
        print(f"\n{'='*50}")
        print(f"Training {agent_name.upper()}")
        print(f"{'='*50}")
        
        rewards, sharpes, pnl = train(
            agent_name, episodes, pricing_model, transaction_cost_model, 
            save_model=f"models/{agent_name}_{pricing_model}_{transaction_cost_model}.pth",
            use_real_data=use_real_data, symbol=symbol, start_date=start_date, end_date=end_date
        )
        results[agent_name] = (rewards, sharpes, pnl)
    
    # Plot comparison
    plot_comparison(results, title=f'Agent Comparison - {pricing_model} - {transaction_cost_model}')
    
    return results

def run_multi_agent_backtest(agents, use_real_data=False, symbol='AAPL', start_date='2023-01-01',
                           end_date=None, option_strike=None, option_expiry=None, 
                           transaction_cost_model='linear', episodes=1, save_plots=None):
    """Run backtest comparison for multiple agents"""
    
    # Choose environment
    if use_real_data:
        env = RealDataDeltaHedgingEnv(
            symbol=symbol, start_date=start_date, end_date=end_date,
            option_strike=option_strike, option_expiry=option_expiry,
            transaction_cost_model=transaction_cost_model
        )
    else:
        env = DeltaHedgingEnv(transaction_cost_model=transaction_cost_model)
    
    # Create agents
    agent_instances = {}
    for agent_name in agents:
        state_dim = env.observation_space.shape[0]
        action_dim = 5 if agent_name == 'dqn' else env.action_space.shape[0]
        agent_instances[agent_name] = AGENT_MAP[agent_name](state_dim, action_dim)
    
    # Run multi-agent backtest
    multi_backtest = MultiAgentBacktest(env, agent_instances)
    results = multi_backtest.run_comparison(episodes)
    
    # Generate comparison analysis
    multi_backtest.plot_comparison(save_path=save_plots)
    multi_backtest.generate_comparison_report()
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo', 'actor_critic'], default='dqn')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save-model', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--pricing-model', type=str, choices=['black_scholes', 'heston', 'merton'], 
                       default='black_scholes', help='Underlying price process model')
    parser.add_argument('--transaction-cost-model', type=str, 
                       choices=['linear', 'nonlinear', 'spread'], default='linear',
                       help='Transaction cost model')
    parser.add_argument('--compare', action='store_true', help='Compare all agents')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate, no training')
    
    # Real data options
    parser.add_argument('--use-real-data', action='store_true', help='Use real market data')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol for real data')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date for real data')
    parser.add_argument('--end-date', type=str, default=None, help='End date for real data')
    parser.add_argument('--option-strike', type=float, default=None, help='Option strike price')
    parser.add_argument('--option-expiry', type=str, default=None, help='Option expiration date')
    
    # Backtesting options
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of training')
    parser.add_argument('--multi-backtest', action='store_true', help='Run multi-agent backtest')
    parser.add_argument('--save-plots', type=str, default=None, help='Path to save plots')
    
    args = parser.parse_args()
    
    if args.backtest:
        backtest(args.agent, args.use_real_data, args.symbol, args.start_date, 
                args.end_date, args.option_strike, args.option_expiry, 
                args.transaction_cost_model, args.load_model, args.episodes, args.save_plots)
    elif args.multi_backtest:
        run_multi_agent_backtest(['dqn', 'ppo', 'actor_critic'], args.use_real_data, 
                               args.symbol, args.start_date, args.end_date,
                               args.option_strike, args.option_expiry, 
                               args.transaction_cost_model, args.episodes, args.save_plots)
    elif args.compare:
        compare_agents(['dqn', 'ppo', 'actor_critic'], args.episodes, 
                      args.pricing_model, args.transaction_cost_model,
                      args.use_real_data, args.symbol, args.start_date, args.end_date)
    else:
        train(args.agent, args.episodes, args.pricing_model, args.transaction_cost_model,
              args.save_model, args.load_model, args.evaluate_only,
              args.use_real_data, args.symbol, args.start_date, args.end_date,
              args.option_strike, args.option_expiry) 