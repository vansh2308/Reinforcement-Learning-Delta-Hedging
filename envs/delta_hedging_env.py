import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DeltaHedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, T=1.0, N=100, S0=100, K=100, r=0.01, sigma=0.2, option_type='call', transaction_cost=0.001, transaction_cost_model='linear', pricing_model='black_scholes', heston_params=None, merton_params=None):
        super(DeltaHedgingEnv, self).__init__()
        self.T = T  # Time to maturity
        self.N = N  # Number of time steps
        self.dt = T / N
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.transaction_cost = transaction_cost
        self.transaction_cost_model = transaction_cost_model
        self.pricing_model = pricing_model
        self.heston_params = heston_params or {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.1, 'rho': -0.7}
        self.merton_params = merton_params or {'lambda': 0.1, 'muJ': -0.1, 'sigmaJ': 0.2}
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Change in hedge position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.t = 0
        self.S = self.S0
        self.hedge = 0.0
        self.cash = 0.0
        self.done = False
        self.price_path = self._simulate_price_path()
        self.option_price_path = [self._bs_price(self.S0, 0)]
        self.pnl = 0.0
        self.pnl_history = []
        return self._get_obs(), {}

    def step(self, action):
        prev_hedge = self.hedge
        action = np.clip(action[0], -1, 1)
        self.hedge += action  # Update hedge position
        transaction_cost = self._compute_transaction_cost(action)
        self.cash -= action * self.S + transaction_cost
        self.t += 1
        self.S = self.price_path[self.t]
        option_price = self._bs_price(self.S, self.t * self.dt)
        self.option_price_path.append(option_price)
        reward = self.hedge * (self.price_path[self.t] - self.price_path[self.t-1]) - transaction_cost
        self.pnl += reward
        self.pnl_history.append(self.pnl)
        if self.t == self.N:
            # Option payoff at maturity
            payoff = max(self.S - self.K, 0) if self.option_type == 'call' else max(self.K - self.S, 0)
            reward += -payoff + self.hedge * (self.S - self.price_path[self.t-1])
            self.done = True
        return self._get_obs(), reward, self.done, False, {}

    def _compute_transaction_cost(self, action):
        """Compute transaction cost based on the selected model"""
        if self.transaction_cost_model == 'linear':
            return abs(action) * self.S * self.transaction_cost
        elif self.transaction_cost_model == 'nonlinear':
            # Nonlinear cost: higher cost for larger trades
            base_cost = abs(action) * self.S * self.transaction_cost
            nonlinear_factor = 1 + 0.5 * abs(action)  # Cost increases with trade size
            return base_cost * nonlinear_factor
        elif self.transaction_cost_model == 'spread':
            # Spread-based cost: bid-ask spread simulation
            spread = self.S * 0.002  # 0.2% spread
            return abs(action) * spread + abs(action) * self.S * self.transaction_cost
        else:
            raise ValueError(f"Unknown transaction cost model: {self.transaction_cost_model}")

    def _get_obs(self):
        delta = self._bs_delta(self.S, self.t * self.dt)
        gamma = self._bs_gamma(self.S, self.t * self.dt)
        obs = np.array([
            self.S,
            self.T - self.t * self.dt,
            self.hedge,
            delta,
            gamma
        ], dtype=np.float32)
        return obs

    def _simulate_price_path(self):
        if self.pricing_model == 'black_scholes':
            return self._simulate_bs_path()
        elif self.pricing_model == 'heston':
            return self._simulate_heston_path()
        elif self.pricing_model == 'merton':
            return self._simulate_merton_path()
        else:
            raise ValueError(f"Unknown pricing model: {self.pricing_model}")

    def _simulate_bs_path(self):
        dt = self.dt
        S = np.zeros(self.N + 1)
        S[0] = self.S0
        for t in range(1, self.N + 1):
            z = np.random.randn()
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z)
        return S

    def _simulate_heston_path(self):
        # Heston model: dS = rSdt + sqrt(v)S dW1, dv = kappa(theta-v)dt + xi sqrt(v) dW2
        p = self.heston_params
        dt = self.dt
        S = np.zeros(self.N + 1)
        v = np.zeros(self.N + 1)
        S[0] = self.S0
        v[0] = p['v0']
        for t in range(1, self.N + 1):
            z1 = np.random.randn()
            z2 = np.random.randn()
            z2 = p['rho'] * z1 + np.sqrt(1 - p['rho'] ** 2) * z2
            v[t] = np.abs(v[t-1] + p['kappa'] * (p['theta'] - v[t-1]) * dt + p['xi'] * np.sqrt(v[t-1] * dt) * z2)
            S[t] = S[t-1] * np.exp((self.r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)
        return S

    def _simulate_merton_path(self):
        # Merton jump diffusion: dS = mu S dt + sigma S dW + Jumps
        p = self.merton_params
        dt = self.dt
        S = np.zeros(self.N + 1)
        S[0] = self.S0
        for t in range(1, self.N + 1):
            z = np.random.randn()
            jump = 0
            if np.random.rand() < p['lambda'] * dt:
                jump = np.random.normal(p['muJ'], p['sigmaJ'])
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z + jump)
        return S

    def _bs_price(self, S, t):
        # Black-Scholes price for European option
        T = self.T - t
        if T <= 0:
            return max(S - self.K, 0) if self.option_type == 'call' else max(self.K - S, 0)
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        from scipy.stats import norm
        if self.option_type == 'call':
            price = S * norm.cdf(d1) - self.K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def _bs_delta(self, S, t):
        T = self.T - t
        if T <= 0:
            return 1.0 if (self.option_type == 'call' and S > self.K) else 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        from scipy.stats import norm
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)

    def _bs_gamma(self, S, t):
        T = self.T - t
        if T <= 0:
            return 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        from scipy.stats import norm
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

    def render(self, mode='human'):
        print(f"t={self.t}, S={self.S:.2f}, hedge={self.hedge:.2f}, cash={self.cash:.2f}, PnL={self.pnl:.2f}") 