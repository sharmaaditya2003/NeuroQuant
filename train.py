import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Import our custom environment and data loader
from src.environment import StockTradingEnv
from src.data_loader import load_and_process_data

# --- CONFIGURATION ---
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2023-01-01"  # Train on 2015-2022
TIMESTEPS = 100000       # Learning steps (Higher = Smarter, but takes longer)

def train():
    # 1. Load Data
    print(f"Loading data for {TICKER}...")
    df = load_and_process_data(TICKER, START_DATE, END_DATE)

    # 2. Create the Environment
    # SB3 requires a "Vectorized Environment" (even if just 1 env)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # 3. Initialize the PPO Agent
    # MlpPolicy = Multi-Layer Perceptron (Standard Neural Network for numbers)
    model = PPO("MlpPolicy", env, verbose=1)

    # 4. Train the Agent
    print(f"🧠 Training PPO Agent for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS)
    print("✅ Training Complete!")

    # 5. Save the Model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/ppo_{TICKER}_100k"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}.zip")

if __name__ == "__main__":
    train()