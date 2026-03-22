import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiFactorPortfolioEnv(gym.Env):
    def __init__(self, master_df: pd.DataFrame, feature_cols: list, target_col: str = 'forward_return', commission_rate: float = 0.001):
        super().__init__()
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.commission_rate = commission_rate
        
        # Extract unique dates and tickers
        self.dates = master_df.index.get_level_values('date').unique().sort_values()
        self.tickers = master_df.index.get_level_values('ticker').unique().sort_values()
        
        self.num_stocks = len(self.tickers)
        self.num_features = len(feature_cols)
        self.num_dates = len(self.dates)
        
        # Create full index to ensure alignment (dates x tickers)
        full_index = pd.MultiIndex.from_product([self.dates, self.tickers], names=['date', 'ticker'])
        # Reindex and fill missing with 0
        aligned_df = master_df.reindex(full_index).fillna(0)
        
        # 3D array for features (num_dates, num_stocks, num_features)
        # Reshape: (num_dates * num_stocks, num_features) -> (num_dates, num_stocks, num_features)
        # Note: aligned_df is sorted by date then ticker due to from_product order
        self.features = aligned_df[feature_cols].values.reshape(self.num_dates, self.num_stocks, self.num_features)
        
        # 2D array for returns (num_dates, num_stocks)
        self.returns = aligned_df[target_col].values.reshape(self.num_dates, self.num_stocks)
        
        # Action space: portfolio weights for each stock
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        
        # Observation space: Flattened features + current weights
        obs_dim = (self.num_stocks * self.num_features) + self.num_stocks
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        # Initialize with equal weights (1/N)
        self.current_weights = np.full(self.num_stocks, 1.0 / self.num_stocks, dtype=np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Get feature matrix for current step
        features_step = self.features[self.current_step]
        
        # Flatten features and concatenate with current weights
        obs = np.concatenate([features_step.flatten(), self.current_weights])
        
        return obs.astype(np.float32)

    def step(self, action):
        # Normalize action to sum to 1
        # Clip to 0 to ensure non-negative weights
        action = np.clip(action, 0, None)
        sum_action = np.sum(action)
        
        if sum_action > 1e-8:
            target_weights = action / sum_action
        else:
            # Fallback to equal weights if sum is effectively 0
            target_weights = np.full(self.num_stocks, 1.0 / self.num_stocks, dtype=np.float32)
            
        # Calculate turnover
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        
        # Calculate transaction cost
        tc = turnover * self.commission_rate
        
        # Get current step returns
        asset_returns = self.returns[self.current_step]
        
        # Calculate portfolio return
        port_return = np.dot(target_weights, asset_returns)
        
        # Calculate Reward
        reward = port_return - tc
        
        # Update state
        self.current_weights = target_weights
        self.current_step += 1
        
        # Determine terminated
        terminated = self.current_step >= (self.num_dates - 1)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {
            "portfolio_return": port_return,
            "turnover": turnover,
            "transaction_cost": tc
        }
