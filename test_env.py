from src.data_loader import load_and_process_data
from src.environment import StockTradingEnv

# 1. Load Data
df = load_and_process_data("AAPL", "2023-01-01", "2024-01-01")

# 2. Create the Environment
env = StockTradingEnv(df)

# 3. Test Random Actions
obs, _ = env.reset()
done = False

print("🟢 Starting Random Trading Test...")
while not done:
    action = env.action_space.sample() # Pick a random action (Buy/Sell/Hold)
    obs, reward, done, _, _ = env.step(action)
    
    if done:
        print(f"🔴 Finished! Final Net Worth: ${obs[5] + (obs[6] * obs[0]):.2f}")