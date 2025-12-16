import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd

# Import our custom modules
from src.environment import StockTradingEnv
from src.data_loader import load_and_process_data

# --- CONFIGURATION ---
TICKER = "AAPL"
TEST_START = "2023-01-01"  # Data the AI has NEVER seen
TEST_END = "2024-01-01"
MODEL_PATH = "models/ppo_AAPL_100k"

def test():
    # 1. Load the Test Data (Unseen 2023 data)
    print(f"Loading test data for {TICKER} ({TEST_START} to {TEST_END})...")
    df_test = load_and_process_data(TICKER, TEST_START, TEST_END)

    # 2. Create the Environment
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])

    # 3. Load the Trained Model
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # 4. Run the Simulation
    obs = env.reset()
    done = False
    
    net_worth_history = []
    
    print("🤖 AI is trading...")
    while not done:
        # Ask the AI what to do (Buy/Sell/Hold)
        action, _states = model.predict(obs)
        
        # Take the action
        obs, rewards, done, info = env.step(action)
        
        # Track Net Worth (Note: 'info' comes as a list because of DummyVecEnv)
        # We need to access the environment inside the vector wrapper to get net_worth
        current_net_worth = env.envs[0].net_worth
        net_worth_history.append(current_net_worth)

    # 5. Visualize Results
    print(f"🏁 Final Net Worth: ${net_worth_history[-1]:.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(net_worth_history, label="AI Agent Net Worth", color='blue')
    
    # Add a benchmark line (Buy and Hold) - Simplified
    # Calculate what happened if we just bought 1 share at start and held
    initial_price = df_test.iloc[0]['Close']
    final_price = df_test.iloc[-1]['Close']
    # Start with $10k. Buy max shares.
    shares_buy_hold = 10000 // initial_price
    buy_hold_final = 10000 + (shares_buy_hold * (final_price - initial_price))
    
    plt.axhline(y=10000, color='r', linestyle='--', label="Initial Cash ($10k)")
    plt.axhline(y=buy_hold_final, color='g', linestyle='--', label=f"Buy & Hold Final (${buy_hold_final:.0f})")
    
    plt.title(f"NeuroQuant: AI Trading Performance on {TICKER} (2023)")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test()