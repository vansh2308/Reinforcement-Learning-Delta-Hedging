import argparse
import numpy as np
from envs.delta_hedging_env import DeltaHedgingEnv
import torch
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.actor_critic_agent import ActorCriticAgent
from utils.plot_utils import plot_training_curve, plot_pnl_curve, plot_sharpe_ratio
from utils.metrics import compute_sharpe_ratio, compute_cumulative_pnl

AGENT_MAP = {
    'dqn': DQNAgent,
    'ppo': PPOAgent,
    'actor_critic': ActorCriticAgent
}

def train(agent_name, episodes=200, pricing_model='black_scholes', save_model=None, load_model=None):
    env = DeltaHedgingEnv(pricing_model=pricing_model)
    state_dim = env.observation_space.shape[0]
    action_dim = 5 if agent_name == 'dqn' else env.action_space.shape[0]
    agent = AGENT_MAP[agent_name](state_dim, action_dim)
    if load_model:
        print(f"Loading model from {load_model}")
        agent.load(load_model)
    episode_rewards = []
    sharpe_ratios = []
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
        if agent_name == 'ppo':
            agent.train_step(next_state)
        elif agent_name == 'actor_critic':
            agent.train_step()
        episode_rewards.append(np.sum(rewards))
        sharpe = compute_sharpe_ratio(rewards)
        sharpe_ratios.append(sharpe)
        print(f"Episode {ep+1}/{episodes} | Reward: {episode_rewards[-1]:.2f} | Sharpe: {sharpe:.2f}")
    # Plotting
    plot_training_curve(episode_rewards, title=f'{agent_name.upper()} Training Curve')
    plot_sharpe_ratio(sharpe_ratios, title=f'{agent_name.upper()} Sharpe Ratio')
    plot_pnl_curve(pnl, title=f'{agent_name.upper()} PnL Curve (Last Episode)')
    if save_model:
        print(f"Saving model to {save_model}")
        agent.save(save_model)
    return episode_rewards, sharpe_ratios, pnl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['dqn', 'ppo', 'actor_critic'], default='dqn')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save-model', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--pricing-model', type=str, choices=['black_scholes', 'heston', 'merton'], default='black_scholes', help='Underlying price process model')
    args = parser.parse_args()
    train(args.agent, args.episodes, pricing_model=args.pricing_model, save_model=args.save_model, load_model=args.load_model) 