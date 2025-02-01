#!/usr/bin/env python
"""
Market-Making RL Agent with PPO

This code demonstrates how to build, train, and evaluate a reinforcement learning (RL) agent 
for a simulated market-making task using a custom OpenAI Gym environment and the PPO algorithm.
Each component is encapsulated in its own class with detailed comments to explain each step.
"""

# -----------------------------------------------------------------------------
# Import required libraries and modules.
# -----------------------------------------------------------------------------
import os                # For operating system interactions like file/directory operations.
import random            # For generating random numbers.
import csv               # For reading/writing CSV files.

import gym               # OpenAI Gym for building RL environments.
from gym import spaces   # For defining action and observation spaces.

import numpy as np       # For numerical operations.
import pandas as pd      # For data manipulation.
import matplotlib.pyplot as plt  # For plotting and visualization.

import yfinance as yf    # For downloading historical financial data.
import torch             # Deep learning library required by PPO.

from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging.
from stable_baselines3 import PPO                  # The PPO RL algorithm.
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # For custom logging and model saving.
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # For vectorized environments and normalization.


# -----------------------------------------------------------------------------
# Utility Function: Set a random seed for reproducibility.
# -----------------------------------------------------------------------------
def set_seed(seed: int = 0):
	"""
	Set random seeds for Python, NumPy, and PyTorch.
	This ensures that the results are reproducible.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)


# -----------------------------------------------------------------------------
# LoggerManager: Manages logging using TensorBoard and CSV.
# -----------------------------------------------------------------------------
class LoggerManager:
	def __init__(self, log_dir="logs"):
		"""
		Initialize the logger by setting up directories for logs,
		a TensorBoard writer, and a CSV file to log training metrics.
		"""
		self.log_dir = log_dir
		self.tensorboard_log_dir = os.path.join(self.log_dir, "tensorboard")
		self.csv_log_file = os.path.join(self.log_dir, "training_logs.csv")
		
		# Create directories if they don't exist.
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.tensorboard_log_dir, exist_ok=True)
		
		# Create a TensorBoard writer.
		self.writer = SummaryWriter(self.tensorboard_log_dir)
		
		# Initialize the CSV file with headers.
		with open(self.csv_log_file, "w", newline="") as f:
			writer_csv = csv.writer(f)
			writer_csv.writerow([
				"step", "reward", "policy_gradient_loss", "value_loss",
				"entropy", "std", "inventory", "pnl", "spread_penalty"
			])


# -----------------------------------------------------------------------------
# TrainingLoggingCallback: A custom callback to log training metrics.
# -----------------------------------------------------------------------------
class TrainingLoggingCallback(BaseCallback):
	def __init__(self, logger_manager: LoggerManager, verbose=0):
		"""
		Initialize the callback with a LoggerManager instance.
		This callback logs training metrics after every training step.
		"""
		super(TrainingLoggingCallback, self).__init__(verbose)
		self.logger_manager = logger_manager
		self.step = 0  # A counter for the training steps.

	def _on_step(self) -> bool:
		"""
		Called at every training step.
		Extracts metrics from the model and environment, logs them to TensorBoard and CSV.
		"""
		# Get rewards and other loss values from the training process.
		rewards = self.locals.get("rewards", [])
		loss = self.model.logger.name_to_value  # Dictionary of loss values logged by PPO.
		policy_loss = loss.get("train/policy_gradient_loss", 0)
		value_loss = loss.get("train/value_loss", 0)
		entropy = loss.get("train/entropy_loss", 0)
		std = loss.get("train/std", 0)

		# Get additional info from the environment (like inventory and PnL).
		infos = self.locals.get("infos", [])
		info = infos[-1] if infos else {}
		inventory = info.get("inventory", 0)
		pnl = info.get("pnl", 0)
		spread_penalty = info.get("spread_penalty", 0)

		# Log metrics to TensorBoard for visualization.
		self.logger_manager.writer.add_scalar("Training/Reward", np.mean(rewards), self.step)
		self.logger_manager.writer.add_scalar("Training/Policy Loss", policy_loss, self.step)
		self.logger_manager.writer.add_scalar("Training/Value Loss", value_loss, self.step)
		self.logger_manager.writer.add_scalar("Training/Entropy", entropy, self.step)
		self.logger_manager.writer.add_scalar("Training/Std", std, self.step)
		self.logger_manager.writer.add_scalar("Training/Inventory", inventory, self.step)
		self.logger_manager.writer.add_scalar("Training/PnL", pnl, self.step)
		self.logger_manager.writer.add_scalar("Training/SpreadPenalty", spread_penalty, self.step)

		# Append the same metrics to a CSV file for record keeping.
		with open(self.logger_manager.csv_log_file, "a", newline="") as f:
			writer_csv = csv.writer(f)
			writer_csv.writerow([
				self.step, np.mean(rewards), policy_loss, value_loss,
				entropy, std, inventory, pnl, spread_penalty
			])

		self.step += 1  # Increase the step counter.
		return True  # Return True to indicate that training should continue.


# -----------------------------------------------------------------------------
# StockMarketMakingEnv: Custom Gym environment for the market-making task.
# -----------------------------------------------------------------------------
class StockMarketMakingEnv(gym.Env):
	def __init__(
		self,
		df: pd.DataFrame,
		max_offset: float = 1.0,
		lot_size: float = 100,
		inventory_penalty_coeff: float = 0.001,
		start_index: int = 0,
		end_index: int = None,
		volatility_window: int = 50
	):
		"""
		Initialize the environment with market data and trading parameters.
		- df: DataFrame with historical market data.
		- max_offset: Maximum allowed offset from the market mid-price for quotes.
		- lot_size: Trading lot size (number of shares/contracts per trade).
		- inventory_penalty_coeff: Coefficient to penalize large inventory positions.
		- start_index and end_index: Define the segment of the data to use.
		- volatility_window: Window size for calculating price volatility.
		"""
		super(StockMarketMakingEnv, self).__init__()
		self.df = df.reset_index(drop=True)
		self.volatility_window = volatility_window

		# Set episode boundaries based on the provided data indices.
		if end_index is None:
			end_index = len(self.df) - 1
		self.start_index = start_index
		self.end_index = end_index
		self.max_steps = self.end_index - self.start_index
		if self.max_steps <= 0:
			raise ValueError(f"Invalid episode length: start_index={start_index}, end_index={end_index}")

		# Trading parameters.
		self.max_offset = max_offset
		self.lot_size = lot_size
		self.inventory_penalty_coeff = inventory_penalty_coeff

		# Initialize the environment's trading state.
		self.current_step = 0
		self.current_index = self.start_index
		self.inventory = 0.0  # How many assets the agent holds.
		self.cash = 0.0       # Cash available from trading.
		self.prev_pnl = 0.0   # Previous profit and loss.
		self.spread_penalty = 0.0  # Penalty for setting narrow spreads.

		# Define the action space.
		# The agent chooses two numbers: one for bid offset and one for ask offset.
		self.action_space = spaces.Box(
			low=np.array([-self.max_offset, -self.max_offset], dtype=np.float32),
			high=np.array([self.max_offset, self.max_offset], dtype=np.float32),
			shape=(2,),
			dtype=np.float32
		)

		# Define the observation space.
		# Observations include current inventory, market mid-price, spread, and remaining steps.
		self.observation_space = spaces.Box(
			low=np.array([-np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
			high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
			dtype=np.float32
		)

	def reset(self):
		"""
		Reset the environment to the initial state.
		This is called at the beginning of each episode.
		"""
		self.current_step = 0
		self.current_index = self.start_index
		self.inventory = 0.0
		self.spread_penalty = 0.0
		self.cash = 0.0
		self.prev_pnl = 0.0
		return self._get_observation()

	def get_volatility(self):
		"""
		Compute volatility based on the percentage change in the closing price
		over the defined volatility window.
		"""
		if self.current_index < self.volatility_window:
			return 0.0
		returns = self.df["Close"].pct_change().iloc[self.current_index - self.volatility_window:self.current_index]
		return returns.std()

	def step(self, action):
		"""
		Execute one step in the environment given the agent's action.
		The action is used to adjust the bid and ask quotes.
		"""
		# Get the current market data row.
		row = self.df.iloc[self.current_index]
		mid_price = 0.5 * (row["High"] + row["Low"])
		spread = row["High"] - row["Low"]
		volatility = self.get_volatility()

		# Decode the agent's action into bid and ask offsets.
		bid_offset, ask_offset = action
		# Ensure that the bid price is always positive.
		agent_bid = max(mid_price + bid_offset, 0.01)
		# Ensure that the ask price is always greater than the bid.
		agent_ask = max(mid_price + ask_offset, agent_bid + 0.01)

		# Compute a penalty based on how tight the spread is relative to volatility.
		self.spread_penalty = abs(ask_offset - bid_offset) - (volatility * 2)
		self.spread_penalty = max(self.spread_penalty, 0) * -0.1

		# Simulate order execution using market data.
		fill_bid_amount, fill_ask_amount, fill_bid_price, fill_ask_price = self._simulate_fills(agent_bid, agent_ask, row)

		# Update the inventory and cash balances based on simulated fills.
		self.inventory += fill_bid_amount
		self.cash -= fill_bid_amount * fill_bid_price
		self.inventory -= fill_ask_amount
		self.cash += fill_ask_amount * fill_ask_price

		# Calculate the mark-to-market profit/loss.
		mark_to_market = self.inventory * row["Close"]
		current_pnl = self.cash + mark_to_market

		# Compute the reward from this step.
		# Reward is based on the incremental PnL minus an inventory penalty and the spread penalty.
		incremental_pnl = current_pnl - self.prev_pnl
		inventory_penalty = self.inventory_penalty_coeff * (self.inventory ** 2)
		raw_reward = np.clip(incremental_pnl - inventory_penalty + self.spread_penalty, -10, 10)
		reward = np.tanh(raw_reward / 10.0) * 10.0  # Normalize reward using tanh.

		# Update the previous PnL for the next step.
		self.prev_pnl = current_pnl

		# Move to the next time step.
		self.current_step += 1
		self.current_index += 1
		done = self.current_index >= self.end_index  # Check if the episode is over.

		# Construct the new observation and additional info.
		obs = self._get_observation()
		info = {"pnl": current_pnl, "inventory": self.inventory, "spread_penalty": self.spread_penalty}
		return obs, reward, done, info

	def _simulate_fills(self, agent_bid, agent_ask, row):
		"""
		Simulate whether the agent's bid and ask orders get filled based on the market data.
		Returns the amount filled and the prices at which they are filled.
		"""
		fill_bid_amount = 0.0
		fill_ask_amount = 0.0
		fill_bid_price = agent_bid
		fill_ask_price = agent_ask

		# Simulate bid fill: if agent's bid is higher than the market's high, it's fully filled.
		if agent_bid >= row["High"]:
			fill_bid_amount = self.lot_size
			fill_bid_price = row["High"]
		# Otherwise, simulate a partial fill based on how close the bid is to the market range.
		elif row["Low"] < agent_bid < row["High"]:
			proximity = (agent_bid - row["Low"]) / (row["High"] - row["Low"])
			fill_bid_amount = self.lot_size * proximity * (row["Volume"] / 1e6)
			fill_bid_amount = min(fill_bid_amount, self.lot_size)

		# Simulate ask fill: if agent's ask is lower than the market's low, it's fully filled.
		if agent_ask <= row["Low"]:
			fill_ask_amount = self.lot_size
			fill_ask_price = row["Low"]
		# Otherwise, simulate a partial fill.
		elif row["Low"] < agent_ask < row["High"]:
			proximity = (row["High"] - agent_ask) / (row["High"] - row["Low"])
			fill_ask_amount = self.lot_size * proximity * (row["Volume"] / 1e6)
			fill_ask_amount = min(fill_ask_amount, self.lot_size)

		return fill_bid_amount, fill_ask_amount, fill_bid_price, fill_ask_price

	def _get_observation(self):
		"""
		Build the observation that the agent will see.
		The observation includes:
		  - Current inventory.
		  - The mid-price of the market.
		  - The market spread (difference between high and low prices).
		  - The number of steps remaining in the episode.
		"""
		row = self.df.iloc[self.current_index]
		mid_price = 0.5 * (row["High"] + row["Low"])
		spread = row["High"] - row["Low"]
		steps_remaining = max(self.max_steps - self.current_step, 0)
		return np.array([self.inventory, mid_price, spread, steps_remaining], dtype=np.float32)

	def render(self, mode='human'):
		"""
		Print out a summary of the current state.
		Useful for debugging or understanding what the agent is doing.
		"""
		row = self.df.iloc[self.current_index]
		current_pnl = self.cash + self.inventory * row["Close"]
		print(f"Step: {self.current_step}, Index: {self.current_index}, "
			  f"Inventory: {self.inventory:.2f}, PnL: {current_pnl:.2f}")


# -----------------------------------------------------------------------------
# Benchmark Market Maker Strategies
# These classes represent simple, non-learning market making strategies.
# -----------------------------------------------------------------------------
class FixedSpreadMarketMaker:
	def __init__(self, spread=0.10):
		"""
		A market maker that uses a fixed spread around the mid-price.
		"""
		self.spread = spread

	def get_quotes(self, mid_price):
		# Returns bid and ask prices by subtracting/adding half the spread.
		return mid_price - self.spread / 2, mid_price + self.spread / 2


class VolatilityAdaptiveMarketMaker:
	def __init__(self, base_spread=0.05, volatility_multiplier=1.5):
		"""
		A market maker that adjusts its spread based on market volatility.
		"""
		self.base_spread = base_spread
		self.volatility_multiplier = volatility_multiplier

	def get_quotes(self, mid_price, volatility):
		# Adjusts spread by multiplying with a volatility factor.
		spread = self.base_spread * (1 + self.volatility_multiplier * volatility)
		return mid_price - spread / 2, mid_price + spread / 2


class TWAPMarketMaker:
	def __init__(self, base_spread=0.05):
		"""
		A Time-Weighted Average Price (TWAP) market maker that uses randomness in its spread.
		"""
		self.base_spread = base_spread

	def get_quotes(self, mid_price):
		# Randomly adjusts the spread within a certain range.
		spread = self.base_spread * random.uniform(0.5, 1.5)
		return mid_price - spread / 2, mid_price + spread / 2


# -----------------------------------------------------------------------------
# MarketDataFetcher: Downloads historical stock data from Yahoo Finance.
# -----------------------------------------------------------------------------
class MarketDataFetcher:
	@staticmethod
	def fetch_data(ticker: str, period: str = "4d", interval: str = "1m") -> pd.DataFrame:
		"""
		Download historical market data for a given ticker.
		- ticker: Stock symbol (e.g., "AAPL").
		- period: Duration of historical data (e.g., "4d" for 4 days).
		- interval: Time interval between data points (e.g., "1m" for 1 minute).
		"""
		stock = yf.Ticker(ticker)
		df = stock.history(period=period, interval=interval)
		df.reset_index(inplace=True)
		return df


# -----------------------------------------------------------------------------
# PPOTrainer: Encapsulates the PPO model training process.
# -----------------------------------------------------------------------------
class PPOTrainer:
	def __init__(self, vec_env, model_save_path: str, model_params: dict, callbacks: list):
		"""
		Initializes the PPOTrainer with the training environment, model parameters,
		save path, and training callbacks.
		- vec_env: A vectorized version of the environment.
		- model_save_path: Directory where the model will be saved.
		- model_params: Dictionary of parameters for PPO.
		- callbacks: List of callback functions for training.
		"""
		self.vec_env = vec_env
		self.model_save_path = model_save_path
		self.callbacks = callbacks
		self.model = PPO("MlpPolicy", vec_env, **model_params)

	def train(self, total_timesteps: int):
		"""
		Train the PPO model for a given number of timesteps.
		"""
		self.model.learn(total_timesteps=total_timesteps, callback=self.callbacks)

	def save(self, filename: str):
		"""
		Save the trained model to a file.
		"""
		self.model.save(filename)

	def get_model(self):
		"""
		Return the trained model.
		"""
		return self.model


# -----------------------------------------------------------------------------
# ModelEvaluator: Evaluates the performance of the trained PPO model.
# -----------------------------------------------------------------------------
class ModelEvaluator:
	def __init__(self, model, vec_env):
		"""
		Initialize with the trained model and evaluation environment.
		"""
		self.model = model
		self.vec_env = vec_env

	def evaluate(self, num_episodes: int = 1):
		"""
		Run a specified number of episodes and return the average reward and PnL.
		"""
		rewards = []
		pnls = []
		for ep_idx in range(num_episodes):
			obs = self.vec_env.reset()
			done = False
			episode_reward = 0.0
			iter_count = 0
			while not done:
				if iter_count > 10000:
					break
				# Predict the next action using the trained model.
				action, _ = self.model.predict(obs, deterministic=True)
				obs, reward, done, infos = self.vec_env.step(action)
				episode_reward += reward
				iter_count += 1
			rewards.append(episode_reward)
			if isinstance(infos, list) and len(infos) > 0:
				pnls.append(infos[0].get("pnl", 0.0))
			else:
				pnls.append(0.0)
		return np.mean(rewards), np.mean(pnls), pnls


# -----------------------------------------------------------------------------
# BenchmarkEvaluator: Evaluates simple, fixed-strategy market makers.
# -----------------------------------------------------------------------------
class BenchmarkEvaluator:
	def __init__(self, benchmarks: dict, df_data: pd.DataFrame):
		"""
		Initialize with a dictionary of benchmark market maker strategies
		and the market data (DataFrame) to evaluate on.
		"""
		self.benchmarks = benchmarks
		self.df_data = df_data

	def evaluate(self):
		"""
		Evaluate each benchmark by simulating trades over the entire dataset.
		Returns a dictionary of results and a time series of PnL for plotting.
		"""
		results = {}
		benchmark_pnl_series = {}
		for name, market_maker in self.benchmarks.items():
			pnl_series = []
			pnl, trades = 0, 0
			for index, row in self.df_data.iterrows():
				mid_price = 0.5 * (row["High"] + row["Low"])
				# Calculate volatility only if enough data is available.
				volatility = (self.df_data["Close"].pct_change()
							  .rolling(50).std().iloc[index] if index >= 50 else 0.0)
				# Get quotes from the benchmark strategy.
				if isinstance(market_maker, VolatilityAdaptiveMarketMaker):
					bid, ask = market_maker.get_quotes(mid_price, volatility)
				else:
					bid, ask = market_maker.get_quotes(mid_price)
				# Simulate trade: if bid or ask are executed, update PnL.
				if row["Low"] <= bid:
					pnl += mid_price - bid
					trades += 1
				if row["High"] >= ask:
					pnl += ask - mid_price
					trades += 1
				pnl_series.append(pnl)
			results[name] = {"PnL": pnl, "Trades": trades}
			benchmark_pnl_series[name] = pnl_series if pnl_series else [0] * len(self.df_data)
		return results, benchmark_pnl_series


# -----------------------------------------------------------------------------
# Visualizer: Contains static methods to plot evaluation results.
# -----------------------------------------------------------------------------
class Visualizer:
	@staticmethod
	def plot_ppo_vs_benchmarks(ppo_pnl, benchmark_pnls, labels):
		"""
		Plot a comparison between the PPO agent and the benchmark strategies.
		- ppo_pnl: PnL time series from the PPO agent.
		- benchmark_pnls: Dictionary of PnL time series from benchmarks.
		- labels: List of labels for the benchmark strategies.
		"""
		plt.figure(figsize=(12, 6))
		plt.plot(ppo_pnl, label='PPO Agent', linestyle='-', linewidth=2)
		for label in labels:
			plt.plot(benchmark_pnls[label], label=label, linestyle='--', linewidth=2)
		plt.xlabel("Time Steps")
		plt.ylabel("Cumulative PnL")
		plt.title("PPO Market Maker vs Benchmark Strategies")
		plt.legend()
		plt.show()

	@staticmethod
	def plot_agent_performance(perf_data: dict):
		"""
		Plot various performance metrics for the agent during one episode.
		The perf_data dictionary should contain keys such as:
			- market_mid_prices, agent_bids, agent_asks, inventory_history,
			  pnl_history, volatility_history.
		"""
		plt.figure(figsize=(15, 12))

		# Plot market mid-price and the agent's bid/ask quotes.
		plt.subplot(4, 1, 1)
		plt.plot(perf_data["market_mid_prices"], label='Market Mid Price')
		plt.plot(perf_data["agent_bids"], label='Agent Bid')
		plt.plot(perf_data["agent_asks"], label='Agent Ask')
		plt.legend()
		plt.title('Quotes and Market Prices')

		# Plot the agent's inventory over time.
		plt.subplot(4, 1, 2)
		plt.plot(perf_data["inventory_history"])
		plt.title('Inventory Over Time')

		# Plot the agent's profit and loss (PnL) over time.
		plt.subplot(4, 1, 3)
		plt.plot(perf_data["pnl_history"])
		plt.title('PnL Over Time')

		# Plot the market volatility over time.
		plt.subplot(4, 1, 4)
		plt.plot(perf_data["volatility_history"], color='red', label='Volatility')
		plt.legend()
		plt.title('Market Volatility Over Time')

		plt.tight_layout()
		plt.show()


# -----------------------------------------------------------------------------
# MarketMakerRunner: Orchestrates the overall pipeline.
# -----------------------------------------------------------------------------
class MarketMakerRunner:
	def __init__(self, config: dict):
		"""
		Initialize the runner with a configuration dictionary.
		The config includes parameters such as:
		  - Stock ticker, data periods, trading parameters,
		  - Total timesteps for training, evaluation parameters, etc.
		"""
		self.config = config
		set_seed(0)  # Ensure reproducibility.
		self.data_fetcher = MarketDataFetcher()  # For downloading market data.
		self.logger_manager = LoggerManager(log_dir=config.get("LOG_DIR", "logs"))

	def run(self):
		"""
		Run the complete training and evaluation pipeline.
		Steps:
		  1. Download and prepare market data.
		  2. Create the training environment.
		  3. Train the PPO agent.
		  4. Evaluate the trained model.
		  5. Compare performance with benchmark strategies.
		  6. Visualize the results.
		"""
		# ---------------------------
		# 1. Download Market Data.
		# ---------------------------
		print(f"Fetching historical data for {self.config['STOCK_TICKER']}...")
		df_data = self.data_fetcher.fetch_data(
			ticker=self.config["STOCK_TICKER"],
			period=self.config["DATA_PERIOD"],
			interval=self.config["DATA_INTERVAL"]
		)
		df_data.dropna(inplace=True)  # Remove any missing values.
		print(f"Fetched {len(df_data)} rows of data.")

		# ---------------------------
		# 2. Create the Training Environment.
		# ---------------------------
		env = StockMarketMakingEnv(
			df=df_data,
			max_offset=self.config["MAX_OFFSET"],
			lot_size=self.config["LOT_SIZE"],
			inventory_penalty_coeff=self.config["INVENTORY_PENALTY"],
			start_index=0,
			end_index=len(df_data) - 1
		)
		# Wrap the environment to support multiple instances and normalization.
		vec_env = DummyVecEnv([lambda: env])
		vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

		# ---------------------------
		# 3. Set up PPO Model and Train.
		# ---------------------------
		# Define model parameters for PPO.
		model_params = {
			"verbose": 1,
			"learning_rate": 0.002,
			"clip_range": 0.3,
			"gae_lambda": 0.95,
			"gamma": 0.99,
			"n_steps": 2048,
			"batch_size": 64,
			"ent_coef": 0.1,
			"vf_coef": 0.25,
			"max_grad_norm": 0.3,
			"policy_kwargs": dict(
				net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
				activation_fn=torch.nn.LeakyReLU,
			)
		}

		# Prepare callbacks for saving the model and logging training metrics.
		model_save_path = self.config["MODEL_SAVE_PATH"]
		os.makedirs(model_save_path, exist_ok=True)
		checkpoint_callback = CheckpointCallback(
			save_freq=self.config["MODEL_SAVE_FREQ"],
			save_path=model_save_path,
			name_prefix='ppo_market_maker'
		)
		training_logging_callback = TrainingLoggingCallback(self.logger_manager)
		callbacks = [checkpoint_callback, training_logging_callback]

		# Initialize and train the PPO agent.
		print("Initializing PPO agent...")
		trainer = PPOTrainer(vec_env, model_save_path, model_params, callbacks)
		print("Training the PPO agent...")
		trainer.train(total_timesteps=self.config["TOTAL_TIMESTEPS"])
		trainer.save(os.path.join(model_save_path, "final_ppo_model"))
		vec_env.save(os.path.join(model_save_path, "vec_normalize.pkl"))
		print("Saved PPO model and VecNormalize statistics.")

		# ---------------------------
		# 4. Evaluate the Trained Model.
		# ---------------------------
		print("Fetching evaluation data...")
		eval_df = self.data_fetcher.fetch_data(
			ticker=self.config["STOCK_TICKER"],
			period="8d",  # Use a different period for evaluation.
			interval=self.config["DATA_INTERVAL"]
		)
		eval_df.dropna(inplace=True)
		eval_env = DummyVecEnv([lambda: StockMarketMakingEnv(
			df=eval_df,
			max_offset=self.config["MAX_OFFSET"],
			lot_size=self.config["LOT_SIZE"],
			inventory_penalty_coeff=self.config["INVENTORY_PENALTY"],
			start_index=0,
			end_index=len(eval_df) - 1
		)])
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

		evaluator = ModelEvaluator(trainer.get_model(), eval_env)
		print("Evaluating on a different time period...")
		avg_reward, avg_pnl, ppo_pnls = evaluator.evaluate(num_episodes=self.config["EVAL_EPISODES"])
		print(f"Evaluation Results - Avg Reward: {avg_reward:.2f}, Avg PnL: {avg_pnl:.2f}")

		# ---------------------------
		# 5. Benchmark Evaluation.
		# ---------------------------
		# Define benchmark market-making strategies.
		benchmarks = {
			"Fixed Spread MM": FixedSpreadMarketMaker(spread=0.10),
			"Volatility Adaptive MM": VolatilityAdaptiveMarketMaker(base_spread=0.05, volatility_multiplier=1.5),
			"TWAP MM": TWAPMarketMaker(base_spread=0.05)
		}
		benchmark_evaluator = BenchmarkEvaluator(benchmarks, eval_df)
		results, benchmark_pnls = benchmark_evaluator.evaluate()
		print("Benchmark evaluation results:")
		for k, v in results.items():
			print(f"{k}: {v}")

		# Plot a comparison of PPO vs benchmark performance.
		Visualizer.plot_ppo_vs_benchmarks(ppo_pnls, benchmark_pnls, list(benchmarks.keys()))

		# ---------------------------
		# 6. Visualize a Single Episode's Performance.
		# ---------------------------
		print("\nRunning a single episode for performance visualization:")
		inventory_history = []
		pnl_history = []
		agent_bids = []
		agent_asks = []
		market_mid_prices = []
		volatility_history = []

		obs = eval_env.reset()
		done = False
		iter_count = 0

		# Run the episode until it's done or a maximum number of iterations.
		while not done and iter_count < 10000:
			action, _ = trainer.get_model().predict(obs, deterministic=True)
			obs, reward, done, infos = eval_env.step(action)
			# Record performance metrics from the environment.
			underlying_env = eval_env.envs[0]
			inventory_history.append(underlying_env.inventory)
			pnl_history.append(infos[0].get("pnl", 0.0))
			# Calculate the market mid-price using the latest data row.
			row = underlying_env.df.iloc[underlying_env.current_index - 1]
			mid_price = 0.5 * (row["High"] + row["Low"])
			agent_bids.append(mid_price + action[0][0])
			agent_asks.append(mid_price + action[0][1])
			market_mid_prices.append(mid_price)
			volatility_history.append(underlying_env.get_volatility())
			iter_count += 1

		# Prepare data for plotting.
		perf_data = {
			"market_mid_prices": market_mid_prices,
			"agent_bids": agent_bids,
			"agent_asks": agent_asks,
			"inventory_history": inventory_history,
			"pnl_history": pnl_history,
			"volatility_history": volatility_history
		}
		Visualizer.plot_agent_performance(perf_data)


# -----------------------------------------------------------------------------
# Main entry point: Run the entire pipeline.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
	# Define configuration parameters for the entire process.
	config = {
		"STOCK_TICKER": "AAPL",           # Stock symbol to use.
		"DATA_PERIOD": "4d",              # Duration for training data.
		"DATA_INTERVAL": "1m",            # Time interval between data points.
		"MAX_OFFSET": 1.0,                # Maximum quote offset.
		"LOT_SIZE": 100,                  # Trade lot size.
		"INVENTORY_PENALTY": 0.001,         # Penalty coefficient for inventory.
		"TOTAL_TIMESTEPS": 50000,         # Total training steps for PPO.
		"EVAL_EPISODES": 1,               # Number of evaluation episodes.
		"MODEL_SAVE_FREQ": 10000,         # Frequency (in timesteps) to save the model.
		"MODEL_SAVE_PATH": "./models",    # Directory to save the trained model.
		"LOG_DIR": "logs"                 # Directory to store logs.
	}

	# Create an instance of MarketMakerRunner and execute the pipeline.
	runner = MarketMakerRunner(config)
	runner.run()

	# Note: If you wish to add hyperparameter tuning (e.g., with Optuna),
	# you can create another class or extend this runner.
