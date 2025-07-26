import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealDataLoader:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol, start_date, end_date=None, interval='1d'):
        """Fetch real stock data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now()
        
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Calculate returns and volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_option_chain(self, symbol, expiration_date=None):
        """Get option chain data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            if expiration_date:
                options = ticker.options
                if expiration_date in options:
                    chain = ticker.option_chain(expiration_date)
                    return chain
                else:
                    print(f"Expiration date {expiration_date} not available. Available dates: {options[:5]}")
                    return None
            else:
                # Get next available expiration
                options = ticker.options
                if options:
                    chain = ticker.option_chain(options[0])
                    return chain
                else:
                    print(f"No options available for {symbol}")
                    return None
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return None
    
    def calculate_implied_volatility(self, S, K, T, r, option_price, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        from scipy.stats import norm
        
        def black_scholes(S, K, T, r, sigma, option_type):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            return price
        
        def vega(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S*np.sqrt(T)*norm.pdf(d1)
        
        # Newton-Raphson to find implied volatility
        sigma = 0.3  # Initial guess
        for i in range(100):
            price = black_scholes(S, K, T, r, sigma, option_type)
            diff = option_price - price
            if abs(diff) < 1e-5:
                break
            sigma = sigma + diff / vega(S, K, T, r, sigma)
            sigma = max(0.001, sigma)  # Ensure positive volatility
        
        return sigma
    
    def get_market_data_for_backtest(self, symbol, start_date, end_date=None, 
                                   option_strike=None, option_expiry=None):
        """Get comprehensive market data for backtesting"""
        # Get stock data
        stock_data = self.get_stock_data(symbol, start_date, end_date)
        if stock_data is None:
            return None
        
        # Get risk-free rate (approximate with 3-month Treasury)
        try:
            treasury = yf.Ticker("^IRX")  # 13-week Treasury
            treasury_data = treasury.history(start=start_date, end=end_date)
            if not treasury_data.empty:
                risk_free_rate = treasury_data['Close'].iloc[-1] / 100
            else:
                risk_free_rate = 0.02  # Default 2%
        except:
            risk_free_rate = 0.02
        
        # Get option data if specified
        option_data = None
        if option_strike and option_expiry:
            option_chain = self.get_option_chain(symbol, option_expiry)
            if option_chain:
                # Find closest strike
                calls = option_chain.calls
                puts = option_chain.puts
                
                # Find closest strike price
                call_strikes = calls['strike'].values
                put_strikes = puts['strike'].values
                
                call_idx = np.argmin(np.abs(call_strikes - option_strike))
                put_idx = np.argmin(np.abs(put_strikes - option_strike))
                
                call_data = calls.iloc[call_idx]
                put_data = puts.iloc[put_idx]
                
                option_data = {
                    'call_price': call_data['lastPrice'],
                    'put_price': put_data['lastPrice'],
                    'call_volume': call_data['volume'],
                    'put_volume': put_data['volume'],
                    'call_iv': call_data.get('impliedVolatility', None),
                    'put_iv': put_data.get('impliedVolatility', None),
                    'strike': call_data['strike'],
                    'expiry': option_expiry
                }
        
        return {
            'stock_data': stock_data,
            'risk_free_rate': risk_free_rate,
            'option_data': option_data,
            'symbol': symbol
        }
    
    def create_real_data_environment_params(self, symbol, start_date, end_date=None, 
                                          option_strike=None, option_expiry=None):
        """Create environment parameters from real market data"""
        market_data = self.get_market_data_for_backtest(symbol, start_date, end_date, 
                                                       option_strike, option_expiry)
        
        if market_data is None:
            return None
        
        stock_data = market_data['stock_data']
        
        # Calculate parameters
        S0 = stock_data['Close'].iloc[0]
        K = option_strike if option_strike else S0
        r = market_data['risk_free_rate']
        
        # Calculate historical volatility
        returns = stock_data['Returns'].dropna()
        sigma = returns.std() * np.sqrt(252)
        
        # Get price path
        price_path = stock_data['Close'].values
        
        return {
            'S0': S0,
            'K': K,
            'r': r,
            'sigma': sigma,
            'price_path': price_path,
            'dates': stock_data.index,
            'option_data': market_data['option_data']
        }

class SyntheticDataGenerator:
    """Generate synthetic data for testing when real data is not available"""
    
    @staticmethod
    def generate_heston_paths(S0, T, N, r, v0, kappa, theta, xi, rho, n_paths=1):
        """Generate multiple Heston paths"""
        dt = T / N
        S = np.zeros((n_paths, N + 1))
        v = np.zeros((n_paths, N + 1))
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(1, N + 1):
            z1 = np.random.randn(n_paths)
            z2 = np.random.randn(n_paths)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            v[:, t] = np.abs(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + 
                            xi * np.sqrt(v[:, t-1] * dt) * z2)
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t-1]) * dt + 
                                        np.sqrt(v[:, t-1] * dt) * z1)
        
        return S, v
    
    @staticmethod
    def generate_jump_diffusion_paths(S0, T, N, r, sigma, lambda_jump, mu_jump, sigma_jump, n_paths=1):
        """Generate multiple jump diffusion paths"""
        dt = T / N
        S = np.zeros((n_paths, N + 1))
        S[:, 0] = S0
        
        for t in range(1, N + 1):
            z = np.random.randn(n_paths)
            jumps = np.random.poisson(lambda_jump * dt, n_paths)
            jump_sizes = np.random.normal(mu_jump, sigma_jump, n_paths)
            
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                        sigma * np.sqrt(dt) * z + jumps * jump_sizes)
        
        return S 