# NeuroQuant 🧠📈
### Deep Reinforcement Learning Trading Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Stable%20Baselines3-brightgreen)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📋 Overview
**NeuroQuant** is an autonomous algorithmic trading agent designed to navigate financial markets using **Deep Reinforcement Learning (DRL)**. Built on top of OpenAI's `Gymnasium` and `Stable Baselines3`, the agent utilizes **Proximal Policy Optimization (PPO)** to learn optimal trading strategies.

The model processes historical **OHLCV (Open, High, Low, Close, Volume)** data enriched with technical indicators to make buy/sell/hold decisions, aiming for robust capital preservation and consistent growth.

## ✨ Key Features
* **🤖 Advanced DRL Agent:** Implements the **PPO (Proximal Policy Optimization)** algorithm for stable and efficient policy updates.
* **📊 Custom Trading Environment:** A specialized `Gymnasium` environment that simulates real-world market conditions, transaction costs, and slippage.
* **📈 Technical Indicator Fusion:** Integrates **RSI, MACD, and SMA** directly into the observation space to give the agent market context.
* **🛡️ Risk Management:** Reward hypothesis designed to prioritize **capital preservation** and risk-adjusted returns (Sharpe Ratio optimization).

## 🛠️ Tech Stack
* **Core Language:** Python
* **RL Framework:** Stable Baselines3, Gymnasium
* **Data Analysis:** Pandas, Pandas-TA, NumPy
* **Visualization:** Matplotlib, Plotly (for backtest rendering)

## 🚀 Performance
* **Training Data:** 8 years of historical market data (2015–2023).
* **Testing:** Validated on **200 days** of unseen out-of-sample data.
* **Training Steps:** Converged after **100,000 timesteps**.
* **Outcome:** Demonstrated strong risk aversion during high-volatility periods compared to standard Buy-and-Hold benchmarks.

## 💻 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/NeuroQuant.git](https://github.com/yourusername/NeuroQuant.git)
    cd NeuroQuant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📉 Usage

To train the agent from scratch:
```bash
python train.py --ticker AAPL --timesteps 100000