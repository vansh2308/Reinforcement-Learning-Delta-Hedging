import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.data_loader import RealDataLoader, SyntheticDataGenerator

class RealDataDeltaHedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol='AAPL', start_date='2023-01-01', end_date=None, 
                 option_strike=None, option_expiry=None, transaction_cost=0.001, 
                 transaction_cost_model='linear', use_real_data=True):
        super(RealDataDeltaHedgingEnv, self).__init__()
        
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.option_strike = option_strike
        self.option_expiry = option_expiry
        self.transaction_cost = transaction_cost
        self.transaction_cost_model = transaction_cost_model
        self.use_real_data = use_real_data
        
        # Load data
        if use_real_data:
            self.data_loader = RealDataLoader()
            self.market_params = self.data_loader.create_real_data_environment_params(
                symbol, start_date, end_date, option_strike, option_expiry
            )
            
            if self.market_params is None:
                print("Failed to load real data, falling back to synthetic data")
                self.use_real_data = False
                self._setup_synthetic_data()
            else:
                self._setup_real_data()
        else:
            self._setup_synthetic_data()
        
        # Environment setup
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def _setup_real_data(self):
        """Setup environment with real market data"""
        params = self.market_params
        self.S0 = params['S0']
        self.K = params['K']
        self.r = params['r']
        self.sigma = params['sigma']
        self.price_path = params['price_path']
        self.dates = params['dates']
        self.option_data = params['option_data']
        
        self.N = len(self.price_path) - 1
        self.T = (self.dates[-1] - self.dates[0]).days / 365.0
        self.dt = self.T / self.N
        
        print(f"Real data loaded: {self.symbol}")
        print(f"Period: {self.dates[0].date()} to {self.dates[-1].date()}")
        print(f"S0: {self.S0:.2f}, K: {self.K:.2f}, r: {self.r:.4f}, σ: {self.sigma:.4f}")

    def _setup_synthetic_data(self):
        """Setup environment with synthetic data"""
        self.S0 = 100
        self.K = 100
        self.r = 0.02
        self.sigma = 0.2
        self.T = 1.0
        self.N = 252  # Daily data
        self.dt = self.T / self.N
        
        # Generate synthetic price path
        generator = SyntheticDataGenerator()
        self.price_path, _ = generator.generate_heston_paths(
            self.S0, self.T, self.N, self.r, 0.04, 2.0, 0.04, 0.1, -0.7, n_paths=1
        )
        self.price_path = self.price_path[0]  # Take first path
        self.dates = None
        self.option_data = None
        
        print("Using synthetic data")

    def reset(self, seed=None, options=None):
        self.t = 0
        self.S = self.price_path[0]
        self.hedge = 0.0
        self.cash = 0.0
        self.done = False
        self.pnl = 0.0
        self.pnl_history = []
        self.option_price_history = []
        
        # Initial option price
        initial_option_price = self._bs_price(self.S, 0)
        self.option_price_history.append(initial_option_price)
        
        return self._get_obs(), {}

    def step(self, action):
        prev_hedge = self.hedge
        action = np.clip(action[0], -1, 1)
        self.hedge += action
        
        # Transaction cost
        transaction_cost = self._compute_transaction_cost(action)
        self.cash -= action * self.S + transaction_cost
        
        # Move to next time step
        self.t += 1
        if self.t <= self.N:
            self.S = self.price_path[self.t]
            option_price = self._bs_price(self.S, self.t * self.dt)
            self.option_price_history.append(option_price)
            
            # Calculate reward (PnL change)
            price_change = self.price_path[self.t] - self.price_path[self.t-1]
            reward = self.hedge * price_change - transaction_cost
            self.pnl += reward
            self.pnl_history.append(self.pnl)
            
            if self.t == self.N:
                # Option payoff at maturity
                payoff = max(self.S - self.K, 0)  # Assuming call option
                reward += -payoff + self.hedge * (self.S - self.price_path[self.t-1])
                self.done = True
        else:
            self.done = True
            reward = 0
        
        return self._get_obs(), reward, self.done, False, {}

    def _compute_transaction_cost(self, action):
        """Compute transaction cost based on the selected model"""
        if self.transaction_cost_model == 'linear':
            return abs(action) * self.S * self.transaction_cost
        elif self.transaction_cost_model == 'nonlinear':
            base_cost = abs(action) * self.S * self.transaction_cost
            nonlinear_factor = 1 + 0.5 * abs(action)
            return base_cost * nonlinear_factor
        elif self.transaction_cost_model == 'spread':
            spread = self.S * 0.002  # 0.2% spread
            return abs(action) * spread + abs(action) * self.S * self.transaction_cost
        else:
            raise ValueError(f"Unknown transaction cost model: {self.transaction_cost_model}")

    def _get_obs(self):
        delta = self._bs_delta(self.S, self.t * self.dt)
        gamma = self._bs_gamma(self.S, self.t * self.dt)
        
        # Add time to expiry as percentage
        time_to_expiry = (self.T - self.t * self.dt) / self.T
        
        obs = np.array([
            self.S,
            time_to_expiry,
            self.hedge,
            delta,
            gamma,
            self.pnl
        ], dtype=np.float32)
        return obs

    def _bs_price(self, S, t):
        """Black-Scholes price for European call option"""
        T = self.T - t
        if T <= 0:
            return max(S - self.K, 0)
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        
        from scipy.stats import norm
        price = S * norm.cdf(d1) - self.K * np.exp(-self.r * T) * norm.cdf(d2)
        return price

    def _bs_delta(self, S, t):
        """Black-Scholes delta"""
        T = self.T - t
        if T <= 0:
            return 1.0 if S > self.K else 0.0
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        from scipy.stats import norm
        return norm.cdf(d1)

    def _bs_gamma(self, S, t):
        """Black-Scholes gamma"""
        T = self.T - t
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        from scipy.stats import norm
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

    def render(self, mode='human'):
        current_date = self.dates[self.t] if self.dates is not None else f"t={self.t}"
        print(f"Date: {current_date}, S: {self.S:.2f}, Hedge: {self.hedge:.2f}, "
              f"Cash: {self.cash:.2f}, PnL: {self.pnl:.2f}")

    def get_market_info(self):
        """Get market information for analysis"""
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'S0': self.S0,
            'K': self.K,
            'r': self.r,
            'sigma': self.sigma,
            'T': self.T,
            'N': self.N,
            'price_path': self.price_path,
            'dates': self.dates,
            'option_data': self.option_data,
            'use_real_data': self.use_real_data
        } 