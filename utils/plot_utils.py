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

def plot_var_cvar(returns, confidence_level=0.95, title='VaR and CVaR Analysis'):
    """Plot returns distribution with VaR and CVaR lines"""
    plt.figure(figsize=(12, 6))
    
    # Plot returns histogram
    plt.hist(returns, bins=50, alpha=0.7, density=True, label='Returns Distribution')
    
    # Compute and plot VaR and CVaR
    from utils.metrics import compute_var, compute_cvar
    var = compute_var(returns, confidence_level)
    cvar = compute_cvar(returns, confidence_level)
    
    # Plot VaR line
    plt.axvline(var, color='red', linestyle='--', linewidth=2, 
                label=f'VaR ({confidence_level*100:.0f}%) = {var:.4f}')
    
    # Plot CVaR line
    plt.axvline(cvar, color='orange', linestyle='--', linewidth=2,
                label=f'CVaR ({confidence_level*100:.0f}%) = {cvar:.4f}')
    
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_drawdown(pnl_series, title='Drawdown Analysis'):
    """Plot drawdown series"""
    from utils.metrics import compute_drawdown
    
    drawdown, max_drawdown = compute_drawdown(pnl_series)
    
    plt.figure(figsize=(12, 8))
    
    # Plot PnL and drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # PnL curve
    ax1.plot(pnl_series, label='Cumulative PnL', color='blue')
    ax1.set_ylabel('PnL')
    ax1.set_title('PnL and Drawdown Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown curve
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.axhline(y=max_drawdown, color='darkred', linestyle='--', 
                label=f'Max Drawdown = {max_drawdown:.4f}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Drawdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_risk_metrics(returns, pnl_series, title='Risk Metrics Summary'):
    """Plot comprehensive risk metrics"""
    from utils.metrics import (compute_sharpe_ratio, compute_var, compute_cvar, 
                              compute_drawdown, compute_volatility, compute_sortino_ratio)
    
    # Compute all metrics
    sharpe = compute_sharpe_ratio(returns)
    var_95 = compute_var(returns, 0.95)
    cvar_95 = compute_cvar(returns, 0.95)
    var_99 = compute_var(returns, 0.99)
    cvar_99 = compute_cvar(returns, 0.99)
    drawdown, max_drawdown = compute_drawdown(pnl_series)
    volatility = compute_volatility(returns)
    sortino = compute_sortino_ratio(returns)
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Returns distribution with VaR/CVaR
    ax1.hist(returns, bins=30, alpha=0.7, density=True)
    ax1.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.4f}')
    ax1.axvline(cvar_95, color='orange', linestyle='--', label=f'CVaR 95%: {cvar_95:.4f}')
    ax1.set_title('Returns Distribution with Risk Measures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.axhline(y=max_drawdown, color='darkred', linestyle='--', 
                label=f'Max DD: {max_drawdown:.4f}')
    ax2.set_title('Drawdown Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Risk metrics bar chart
    metrics = ['Sharpe', 'Sortino', 'Volatility', 'Max DD']
    values = [sharpe, sortino, volatility, abs(max_drawdown)]
    colors = ['green', 'blue', 'orange', 'red']
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_title('Risk Metrics Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # VaR comparison
    var_levels = ['95%', '99%']
    var_values = [abs(var_95), abs(var_99)]
    cvar_values = [abs(cvar_95), abs(cvar_99)]
    
    x = np.arange(len(var_levels))
    width = 0.35
    
    ax4.bar(x - width/2, var_values, width, label='VaR', alpha=0.7)
    ax4.bar(x + width/2, cvar_values, width, label='CVaR', alpha=0.7)
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Risk Measure')
    ax4.set_title('VaR vs CVaR')
    ax4.set_xticks(x)
    ax4.set_xticklabels(var_levels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Risk Metrics Summary:")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"VaR (95%): {var_95:.4f}")
    print(f"CVaR (95%): {cvar_95:.4f}")
    print(f"VaR (99%): {var_99:.4f}")
    print(f"CVaR (99%): {cvar_99:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")

def plot_comparison(agents_results, title='Agent Comparison'):
    """Plot comparison of different agents"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for agent_name, results in agents_results.items():
        rewards, sharpes, pnl = results
        
        # Training curves
        ax1.plot(rewards, label=f'{agent_name} Rewards', alpha=0.7)
        
        # Sharpe ratios
        ax2.plot(sharpes, label=f'{agent_name} Sharpe', alpha=0.7)
        
        # Final PnL
        ax3.plot(pnl, label=f'{agent_name} PnL', alpha=0.7)
    
    ax1.set_title('Training Curves')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Sharpe Ratios')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('PnL Curves (Last Episode)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('PnL')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    agent_names = list(agents_results.keys())
    final_rewards = [agents_results[name][0][-1] for name in agent_names]
    final_sharpes = [agents_results[name][1][-1] for name in agent_names]
    
    x = np.arange(len(agent_names))
    width = 0.35
    
    ax4.bar(x - width/2, final_rewards, width, label='Final Reward', alpha=0.7)
    ax4.bar(x + width/2, final_sharpes, width, label='Final Sharpe', alpha=0.7)
    ax4.set_xlabel('Agent')
    ax4.set_ylabel('Value')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agent_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show() 