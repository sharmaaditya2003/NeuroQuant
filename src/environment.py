import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """
    A custom Stock Trading Environment that follows gym interface.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        
        # Action Space: 3 options [0: Hold, 1: Buy, 2: Sell]
        self.action_space = spaces.Discrete(3)

        # Observation Space: [Close, RSI, SMA_20, SMA_50, MACD, Balance, Shares_Held]
        # We normalize these values roughly between 0 and infinity (or -1 to 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Initialize State
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df) - 1

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        return self._next_observation(), {}

    def _next_observation(self):
        """Returns the current state of the market + our portfolio"""
        # Get the row of data for the current day
        frame = self.df.iloc[self.current_step]
        
        # Construct the observation vector
        obs = np.array([
            frame['Close'],
            frame['RSI'],
            frame['SMA_20'],
            frame['SMA_50'],
            frame['MACD'],
            self.balance,
            self.shares_held
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # Get current price
        current_price = self.df.iloc[self.current_step]['Close']
        
        # --- EXECUTE ACTION ---
        # 0: Hold (Do nothing)
        # 1: Buy (Buy 1 share - Simplified for now)
        if action == 1 and self.balance >= current_price:
            self.balance -= current_price
            self.shares_held += 1
            
        # 2: Sell (Sell 1 share - Simplified for now)
        elif action == 2 and self.shares_held > 0:
            self.balance += current_price
            self.shares_held -= 1

        # --- CALCULATE REWARD ---
        # New Net Worth = Cash + (Shares * Current Price)
        new_net_worth = self.balance + (self.shares_held * current_price)
        
        # Reward = Change in Net Worth (Did we make money this step?)
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth

        # --- CHECK IF DONE ---
        terminated = self.current_step >= self.max_steps
        truncated = False # Not used for now

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Print the current status"""
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}")