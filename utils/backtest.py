import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from utils.metrics import (compute_sharpe_ratio, compute_var, compute_cvar, 
                          compute_drawdown, compute_volatility, compute_sortino_ratio)
from utils.plot_utils import plot_risk_metrics

class BacktestEngine:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.results = {}
        
    def run_backtest(self, episodes=1):
        """Run backtest for specified number of episodes"""
        all_rewards = []
        all_pnl_series = []
        all_actions = []
        all_states = []
        
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_rewards = []
            episode_pnl = []
            episode_actions = []
            episode_states = []
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                episode_rewards.append(reward)
                episode_pnl.append(self.env.pnl)
                episode_actions.append(action[0])
                episode_states.append(state)
                
                state = next_state
            
            all_rewards.extend(episode_rewards)
            all_pnl_series.append(episode_pnl)
            all_actions.extend(episode_actions)
            all_states.extend(episode_states)
        
        # Store results
        self.results = {
            'rewards': all_rewards,
            'pnl_series': all_pnl_series,
            'actions': all_actions,
            'states': all_states,
            'market_info': self.env.get_market_info()
        }
        
        return self.results
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        rewards = self.results['rewards']
        pnl_series = self.results['pnl_series'][-1] if self.results['pnl_series'] else []
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = np.sum(rewards)
        metrics['mean_return'] = np.mean(rewards)
        metrics['std_return'] = np.std(rewards)
        
        # Risk metrics
        metrics['sharpe_ratio'] = compute_sharpe_ratio(rewards)
        metrics['sortino_ratio'] = compute_sortino_ratio(rewards)
        metrics['volatility'] = compute_volatility(rewards)
        metrics['var_95'] = compute_var(rewards, 0.95)
        metrics['cvar_95'] = compute_cvar(rewards, 0.95)
        metrics['var_99'] = compute_var(rewards, 0.99)
        metrics['cvar_99'] = compute_cvar(rewards, 0.99)
        
        # Drawdown metrics
        if pnl_series:
            drawdown, max_drawdown = compute_drawdown(pnl_series)
            metrics['max_drawdown'] = max_drawdown
            metrics['drawdown_series'] = drawdown
        
        # Trading metrics
        actions = self.results['actions']
        metrics['total_trades'] = len([a for a in actions if abs(a) > 0.01])
        metrics['avg_trade_size'] = np.mean([abs(a) for a in actions if abs(a) > 0.01])
        metrics['win_rate'] = len([r for r in rewards if r > 0]) / len(rewards) if rewards else 0
        
        return metrics
    
    def plot_backtest_results(self, save_path=None):
        """Plot comprehensive backtest results"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        rewards = self.results['rewards']
        pnl_series = self.results['pnl_series'][-1] if self.results['pnl_series'] else []
        actions = self.results['actions']
        market_info = self.results['market_info']
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price and PnL
        ax1 = plt.subplot(3, 3, 1)
        if market_info['dates'] is not None:
            dates = market_info['dates']
            ax1.plot(dates, market_info['price_path'], label='Stock Price', color='blue')
            ax1.set_title('Stock Price Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
        else:
            ax1.plot(market_info['price_path'], label='Stock Price', color='blue')
            ax1.set_title('Stock Price Over Time')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PnL curve
        ax2 = plt.subplot(3, 3, 2)
        if pnl_series:
            ax2.plot(pnl_series, label='Cumulative PnL', color='green')
            ax2.set_title('Cumulative PnL')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('PnL')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Returns distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(rewards, bins=50, alpha=0.7, density=True, color='orange')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Return')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        
        # 4. Action distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(actions, bins=50, alpha=0.7, color='red')
        ax4.set_title('Action Distribution')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe ratio
        ax5 = plt.subplot(3, 3, 5)
        if len(rewards) > 20:
            rolling_sharpe = []
            window = min(20, len(rewards) // 4)
            for i in range(window, len(rewards)):
                rolling_sharpe.append(compute_sharpe_ratio(rewards[i-window:i]))
            ax5.plot(rolling_sharpe, label=f'Rolling Sharpe ({window} periods)', color='purple')
            ax5.set_title('Rolling Sharpe Ratio')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown
        ax6 = plt.subplot(3, 3, 6)
        if pnl_series:
            drawdown, max_drawdown = compute_drawdown(pnl_series)
            ax6.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            ax6.plot(drawdown, color='red', linewidth=1)
            ax6.axhline(y=max_drawdown, color='darkred', linestyle='--', 
                       label=f'Max DD: {max_drawdown:.4f}')
            ax6.set_title('Drawdown Analysis')
            ax6.set_xlabel('Time Step')
            ax6.set_ylabel('Drawdown')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Risk metrics summary
        ax7 = plt.subplot(3, 3, 7)
        metrics = self.calculate_performance_metrics()
        metric_names = ['Sharpe', 'Sortino', 'Volatility', 'Max DD']
        metric_values = [metrics['sharpe_ratio'], metrics['sortino_ratio'], 
                        metrics['volatility'], abs(metrics.get('max_drawdown', 0))]
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax7.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax7.set_title('Risk Metrics Summary')
        ax7.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 8. VaR/CVaR
        ax8 = plt.subplot(3, 3, 8)
        var_95 = metrics['var_95']
        cvar_95 = metrics['cvar_95']
        var_99 = metrics['var_99']
        cvar_99 = metrics['cvar_99']
        
        var_levels = ['95%', '99%']
        var_values = [abs(var_95), abs(var_99)]
        cvar_values = [abs(cvar_95), abs(cvar_99)]
        
        x = np.arange(len(var_levels))
        width = 0.35
        
        ax8.bar(x - width/2, var_values, width, label='VaR', alpha=0.7)
        ax8.bar(x + width/2, cvar_values, width, label='CVaR', alpha=0.7)
        ax8.set_xlabel('Confidence Level')
        ax8.set_ylabel('Risk Measure')
        ax8.set_title('VaR vs CVaR')
        ax8.set_xticks(x)
        ax8.set_xticklabels(var_levels)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Trading statistics
        ax9 = plt.subplot(3, 3, 9)
        trading_stats = ['Total Trades', 'Win Rate', 'Avg Trade Size']
        trading_values = [metrics['total_trades'], metrics['win_rate'], metrics['avg_trade_size']]
        colors = ['cyan', 'magenta', 'yellow']
        bars = ax9.bar(trading_stats, trading_values, color=colors, alpha=0.7)
        ax9.set_title('Trading Statistics')
        ax9.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, trading_values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'Backtest Results - {market_info["symbol"]} ({market_info["start_date"]} to {market_info["end_date"]})', 
                    fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive backtest report"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        metrics = self.calculate_performance_metrics()
        market_info = self.results['market_info']
        
        report = f"""
        ========================================
        BACKTEST REPORT
        ========================================
        
        Market Information:
        - Symbol: {market_info['symbol']}
        - Period: {market_info['start_date']} to {market_info['end_date']}
        - Initial Price: ${market_info['S0']:.2f}
        - Strike Price: ${market_info['K']:.2f}
        - Risk-free Rate: {market_info['r']:.4f}
        - Volatility: {market_info['sigma']:.4f}
        - Data Type: {'Real Market Data' if market_info['use_real_data'] else 'Synthetic Data'}
        
        Performance Metrics:
        - Total Return: {metrics['total_return']:.4f}
        - Mean Return: {metrics['mean_return']:.4f}
        - Standard Deviation: {metrics['std_return']:.4f}
        - Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
        - Sortino Ratio: {metrics['sortino_ratio']:.4f}
        - Volatility: {metrics['volatility']:.4f}
        
        Risk Metrics:
        - VaR (95%): {metrics['var_95']:.4f}
        - CVaR (95%): {metrics['cvar_95']:.4f}
        - VaR (99%): {metrics['var_99']:.4f}
        - CVaR (99%): {metrics['cvar_99']:.4f}
        - Maximum Drawdown: {metrics.get('max_drawdown', 0):.4f}
        
        Trading Statistics:
        - Total Trades: {metrics['total_trades']}
        - Win Rate: {metrics['win_rate']:.2%}
        - Average Trade Size: {metrics['avg_trade_size']:.4f}
        
        ========================================
        """
        
        print(report)
        return report

class MultiAgentBacktest:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.results = {}
        
    def run_comparison(self, episodes=1):
        """Run backtest for multiple agents"""
        comparison_results = {}
        
        for agent_name, agent in self.agents.items():
            print(f"Running backtest for {agent_name}...")
            backtest = BacktestEngine(self.env, agent)
            results = backtest.run_backtest(episodes)
            metrics = backtest.calculate_performance_metrics()
            
            comparison_results[agent_name] = {
                'results': results,
                'metrics': metrics
            }
        
        self.results = comparison_results
        return comparison_results
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of multiple agents"""
        if not self.results:
            raise ValueError("No comparison results available. Run comparison first.")
        
        agent_names = list(self.results.keys())
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sharpe Ratio comparison
        sharpe_ratios = [self.results[name]['metrics']['sharpe_ratio'] for name in agent_names]
        ax1.bar(agent_names, sharpe_ratios, alpha=0.7)
        ax1.set_title('Sharpe Ratio Comparison')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Return comparison
        total_returns = [self.results[name]['metrics']['total_return'] for name in agent_names]
        ax2.bar(agent_names, total_returns, alpha=0.7, color='green')
        ax2.set_title('Total Return Comparison')
        ax2.set_ylabel('Total Return')
        ax2.grid(True, alpha=0.3)
        
        # 3. Maximum Drawdown comparison
        max_drawdowns = [abs(self.results[name]['metrics'].get('max_drawdown', 0)) for name in agent_names]
        ax3.bar(agent_names, max_drawdowns, alpha=0.7, color='red')
        ax3.set_title('Maximum Drawdown Comparison')
        ax3.set_ylabel('Max Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # 4. PnL curves comparison
        for name in agent_names:
            pnl_series = self.results[name]['results']['pnl_series'][-1] if self.results[name]['results']['pnl_series'] else []
            if pnl_series:
                ax4.plot(pnl_series, label=name, alpha=0.7)
        
        ax4.set_title('PnL Curves Comparison')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Cumulative PnL')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Agent Backtest Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def generate_comparison_report(self):
        """Generate comparison report for all agents"""
        if not self.results:
            raise ValueError("No comparison results available. Run comparison first.")
        
        report = "========================================\n"
        report += "MULTI-AGENT BACKTEST COMPARISON REPORT\n"
        report += "========================================\n\n"
        
        # Create comparison table
        metrics_to_compare = ['total_return', 'sharpe_ratio', 'sortino_ratio', 
                             'volatility', 'max_drawdown', 'win_rate']
        
        report += f"{'Agent':<15} {'Return':<10} {'Sharpe':<10} {'Sortino':<10} {'Vol':<8} {'MaxDD':<8} {'Win%':<8}\n"
        report += "-" * 80 + "\n"
        
        for agent_name in self.results.keys():
            metrics = self.results[agent_name]['metrics']
            report += f"{agent_name:<15} "
            report += f"{metrics['total_return']:<10.4f} "
            report += f"{metrics['sharpe_ratio']:<10.4f} "
            report += f"{metrics['sortino_ratio']:<10.4f} "
            report += f"{metrics['volatility']:<8.4f} "
            report += f"{abs(metrics.get('max_drawdown', 0)):<8.4f} "
            report += f"{metrics['win_rate']:<8.2%}\n"
        
        report += "\n========================================\n"
        
        print(report)
        return report 