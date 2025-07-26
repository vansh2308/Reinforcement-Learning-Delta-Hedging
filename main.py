import argparse
import numpy as np
from envs.delta_hedging_env import DeltaHedgingEnv
import torch
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.actor_critic_agent import ActorCriticAgent
from utils.plot_utils import (plot_training_curve, plot_pnl_curve, plot_sharpe_ratio, 
                             plot_var_cvar, plot_drawdown, plot_risk_metrics, plot_comparison)
from utils.metrics import (compute_sharpe_ratio, compute_cumulative_pnl, compute_var, 
                          compute_cvar, compute_drawdown, compute_volatility, compute_sortino_ratio)

AGENT_MAP = {
    'dqn': DQNAgent,
    'ppo': PPOAgent,
    'actor_critic': ActorCriticAgent
}

def train(agent_name, episodes=200, pricing_model='black_scholes', transaction_cost_model='linear', 
          save_model=None, load_model=None, evaluate_only=False):
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

def compare_agents(agents, episodes=200, pricing_model='black_scholes', transaction_cost_model='linear'):
    """Compare multiple agents"""
    results = {}
    
    for agent_name in agents:
        print(f"\n{'='*50}")
        print(f"Training {agent_name.upper()}")
        print(f"{'='*50}")
        
        rewards, sharpes, pnl = train(
            agent_name, episodes, pricing_model, transaction_cost_model, 
            save_model=f"models/{agent_name}_{pricing_model}_{transaction_cost_model}.pth"
        )
        results[agent_name] = (rewards, sharpes, pnl)
    
    # Plot comparison
    plot_comparison(results, title=f'Agent Comparison - {pricing_model} - {transaction_cost_model}')
    
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
    
    args = parser.parse_args()
    
    if args.compare:
        compare_agents(['dqn', 'ppo', 'actor_critic'], args.episodes, 
                      args.pricing_model, args.transaction_cost_model)
    else:
        train(args.agent, args.episodes, args.pricing_model, args.transaction_cost_model,
              args.save_model, args.load_model, args.evaluate_only) 