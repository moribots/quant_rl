# Import required libraries
import os  # Operating system interfaces
import yfinance as yf  # Financial data download
import gym  # Reinforcement learning environments
import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Data visualization
from gym import spaces  # RL action/observation space definitions
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # Vectorized environments
import torch  # Deep learning framework
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # Training utilities
import random  # Random number generation
import csv  # CSV file handling
from torch.utils.tensorboard import SummaryWriter  # Training visualization

# Set up logging directories
LOG_DIR = "logs"
CSV_LOG_FILE = os.path.join(LOG_DIR, "training_logs.csv")
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")
# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Set random seed for reproducibility across multiple libraries
def set_seed(seed: int = 0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

set_seed(0)  # Initialize all random seeds

# TensorBoard Writer for logging training metrics
writer = SummaryWriter(TENSORBOARD_LOG_DIR)

# CSV Logging setup - create file and write header
with open(CSV_LOG_FILE, "w", newline="") as f:
	writer_csv = csv.writer(f)
	writer_csv.writerow(["step", "reward", "policy_gradient_loss", "value_loss", "entropy", "inventory", "pnl"])

# Custom callback for logging training metrics to both TensorBoard and CSV
class TrainingLoggingCallback(BaseCallback):
	def __init__(self, verbose=0):
		super(TrainingLoggingCallback, self).__init__(verbose)
		self.step = 0  # Initialize step counter

	def _on_step(self) -> bool:
		# Extract training metrics from logger
		rewards = self.locals["rewards"]
		loss = self.model.logger.name_to_value
		policy_loss = loss.get("train/policy_gradient_loss", -1)
		value_loss = loss.get("train/value_loss", -1)
		entropy = loss.get("train/entropy_loss", -1)
		
		# Get environment information
		info = self.locals["infos"][-1]  # Latest episode info
		inventory = info.get("inventory", -1)
		pnl = info.get("pnl", -1)
		
		# Log metrics to TensorBoard
		writer.add_scalar("Training/Reward", np.mean(rewards), self.step)
		writer.add_scalar("Training/Policy Loss", policy_loss, self.step)
		writer.add_scalar("Training/Value Loss", value_loss, self.step)
		writer.add_scalar("Training/Entropy", entropy, self.step)
		writer.add_scalar("Training/Inventory", inventory, self.step)
		writer.add_scalar("Training/PnL", pnl, self.step)
		
		# Append metrics to CSV file
		with open(CSV_LOG_FILE, "a", newline="") as f:
			writer_csv = csv.writer(f)
			writer_csv.writerow([self.step, np.mean(rewards), policy_loss, value_loss, entropy, inventory, pnl])
		
		self.step += 1  # Increment step counter
		return True  # Continue training

# Custom Gym environment for stock market making
class StockMarketMakingEnv(gym.Env):
	def __init__(
		self,
		df: pd.DataFrame,
		max_offset: float = 1.0,
		lot_size: float = 100,
		inventory_penalty_coeff: float = 0.001,
		start_index: int = 0,
		end_index: int = None
	):
		super(StockMarketMakingEnv, self).__init__()

		# Initialize market data
		self.df = df.reset_index(drop=True)

		# Set episode boundaries
		if end_index is None:
			end_index = len(self.df) - 1
		self.start_index = start_index
		self.end_index = end_index
		self.max_steps = self.end_index - self.start_index

		if self.max_steps <= 0:
			raise ValueError(f"Invalid episode length: start_index={start_index}, end_index={end_index}")

		# Set trading parameters
		self.max_offset = max_offset
		self.lot_size = lot_size
		self.inventory_penalty_coeff = inventory_penalty_coeff

		# Initialize trading state
		self.current_step = 0
		self.current_index = self.start_index
		self.inventory = 0.0  # Current asset holdings
		self.cash = 0.0  # Available funds
		self.prev_pnl = 0.0  # Previous profit/loss

		# Define action space: [bid_offset, ask_offset] from mid price
		self.action_space = spaces.Box(
			low=np.array([-self.max_offset, -self.max_offset], dtype=np.float32),
			high=np.array([self.max_offset, self.max_offset], dtype=np.float32),
			shape=(2,),
			dtype=np.float32
		)

		# Define observation space: [inventory, mid_price, spread, steps_remaining]
		self.observation_space = spaces.Box(
			low=np.array([-np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
			high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
			dtype=np.float32
		)

	def reset(self):
		"""Reset environment to initial state"""
		self.current_step = 0
		self.current_index = self.start_index
		self.inventory = 0.0
		self.cash = 0.0
		self.prev_pnl = 0.0
		return self._get_observation()

	def step(self, action):
		"""Execute one timestep of the environment"""
		row = self.df.iloc[self.current_index]
		mid_price = 0.5 * (row["High"] + row["Low"])
		spread = row["High"] - row["Low"]

		# Decode action into bid/ask prices
		bid_offset, ask_offset = action
		agent_bid = max(mid_price + bid_offset, 0.01)  # Ensure positive prices
		agent_ask = max(mid_price + ask_offset, agent_bid + 0.01)  # Ensure ask > bid

		# Simulate order execution
		fill_bid_amount, fill_ask_amount, fill_bid_price, fill_ask_price = self._simulate_fills(
			agent_bid, agent_ask, row
		)

		# Update inventory and cash based on fills
		self.inventory += fill_bid_amount
		self.cash -= fill_bid_amount * fill_bid_price
		self.inventory -= fill_ask_amount
		self.cash += fill_ask_amount * fill_ask_price

		# Calculate PnL
		mark_to_market = self.inventory * row["Close"]
		current_pnl = self.cash + mark_to_market

		# Calculate reward components
		incremental_pnl = current_pnl - self.prev_pnl
		normalized_pnl = incremental_pnl
		inventory_penalty = self.inventory_penalty_coeff * (self.inventory ** 2)
		reward = np.clip(normalized_pnl - inventory_penalty, -10, 10)

		self.prev_pnl = current_pnl  # Store for next step

		# Update environment state
		self.current_step += 1
		self.current_index += 1
		done = self.current_index >= self.end_index  # Episode completion check

		obs = self._get_observation()
		info = {"pnl": current_pnl, "inventory": self.inventory}

		return obs, reward, done, info

	def _simulate_fills(self, agent_bid, agent_ask, row):
		"""Simulate order execution based on market conditions"""
		fill_bid_amount = 0.0
		fill_ask_amount = 0.0
		fill_bid_price = agent_bid
		fill_ask_price = agent_ask

		# Calculate bid fill amount based on price and market volume
		if agent_bid >= row["High"]:
			# Full fill if bid is above market high
			fill_bid_amount = self.lot_size
			fill_bid_price = row["High"]
		elif row["Low"] < agent_bid < row["High"]:
			# Partial fill based on price proximity to market range
			proximity = (agent_bid - row["Low"]) / (row["High"] - row["Low"])
			fill_bid_amount = self.lot_size * proximity * (row["Volume"] / 1e6)
			fill_bid_amount = min(fill_bid_amount, self.lot_size)

		# Calculate ask fill amount based on price and market volume
		if agent_ask <= row["Low"]:
			# Full fill if ask is below market low
			fill_ask_amount = self.lot_size
			fill_ask_price = row["Low"]
		elif row["Low"] < agent_ask < row["High"]:
			# Partial fill based on price proximity to market range
			proximity = (row["High"] - agent_ask) / (row["High"] - row["Low"])
			fill_ask_amount = self.lot_size * proximity * (row["Volume"] / 1e6)
			fill_ask_amount = min(fill_ask_amount, self.lot_size)

		return fill_bid_amount, fill_ask_amount, fill_bid_price, fill_ask_price

	def _get_observation(self):
		"""Construct current environment observation"""
		row = self.df.iloc[self.current_index]
		mid_price = 0.5 * (row["High"] + row["Low"])
		spread = row["High"] - row["Low"]
		steps_remaining = self.max_steps - self.current_step
		steps_remaining = max(steps_remaining, 0)

		return np.array([
			self.inventory,
			mid_price,
			spread,
			steps_remaining
		], dtype=np.float32)

	def render(self, mode='human'):
		"""Display current environment state"""
		row = self.df.iloc[self.current_index]
		current_pnl = self.cash + self.inventory * row["Close"]
		print(
			f"Step: {self.current_step}, Index: {self.current_index}, "
			f"Inventory: {self.inventory:.2f}, PnL: {current_pnl:.2f}"
		)

def evaluate_model(model, vec_env, num_episodes=1):
	"""Evaluate trained model performance"""
	rewards = []
	pnls = []
	for _ in range(num_episodes):
		obs = vec_env.reset()
		done = False
		episode_reward = 0.0
		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, infos = vec_env.step(action)
			episode_reward += reward
		rewards.append(episode_reward)
		if isinstance(infos, list) and len(infos) > 0:
			pnls.append(infos[0].get("pnl", 0.0))
		else:
			pnls.append(0.0)
	return np.mean(rewards), np.mean(pnls)

def fetch_historical_stock_data(ticker: str, period: str = "4d", interval: str = "1m"):
	"""Download historical stock data using Yahoo Finance"""
	stock = yf.Ticker(ticker)
	df = stock.history(period=period, interval=interval)
	df.reset_index(inplace=True)
	return df

def objective(trial):
	"""Objective function for hyperparameter optimization with Optuna"""
	# Suggest hyperparameters
	learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
	ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 0.1)
	clip_range = trial.suggest_uniform("clip_range", 0.1, 0.4)
	
	# Create and train model with suggested parameters
	model = PPO(
		"MlpPolicy",
		DummyVecEnv([lambda: StockMarketMakingEnv(df=fetch_evaluation_data("AAPL", "1mo", "1m"), max_offset=1.0, lot_size=100, inventory_penalty_coeff=0.001, start_index=0, end_index=1000)]),
		learning_rate=learning_rate,
		ent_coef=ent_coef,
		clip_range=clip_range,
		verbose=0
	)
	model.learn(total_timesteps=50000)
	
	# Evaluate model performance
	eval_df = fetch_evaluation_data(ticker="AAPL", period="8d", interval="1m")
	eval_env = DummyVecEnv([lambda: StockMarketMakingEnv(df=eval_df, max_offset=1.0, lot_size=100, inventory_penalty_coeff=0.001, start_index=0, end_index=len(eval_df) - 1)])
	eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
	
	avg_reward, avg_pnl = evaluate_model(model, eval_env, num_episodes=5)
	return avg_reward

if __name__ == "__main__":
	# Configuration parameters
	STOCK_TICKER = "AAPL"
	DATA_PERIOD = "4d"
	DATA_INTERVAL = "1m"
	MAX_OFFSET = 1.0
	LOT_SIZE = 100
	INVENTORY_PENALTY = 0.001
	TOTAL_TIMESTEPS = 50000
	EVAL_EPISODES = 1
	MODEL_SAVE_FREQ = 10000
	MODEL_SAVE_PATH = "./models/"

	os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # Create model save directory

	# Fetch and prepare market data
	print(f"Fetching historical data for {STOCK_TICKER}...")
	df_data = fetch_historical_stock_data(
		ticker=STOCK_TICKER,
		period=DATA_PERIOD,
		interval=DATA_INTERVAL
	)
	df_data.dropna(inplace=True)  # Remove missing values
	print(f"Fetched {len(df_data)} rows of data.")

	# Initialize trading environment
	print("Creating the market-making environment...")
	env = StockMarketMakingEnv(
		df=df_data,
		max_offset=MAX_OFFSET,
		lot_size=LOT_SIZE,
		inventory_penalty_coeff=INVENTORY_PENALTY,
		start_index=0,
		end_index=len(df_data) - 1
	)

	# Setup vectorized and normalized environment
	print("Wrapping the environment with DummyVecEnv and VecNormalize...")
	vec_env = DummyVecEnv([lambda: env])
	vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

	# Initialize PPO model with specific architecture
	print("Initializing PPO agent...")
	model = PPO(
		"MlpPolicy",
		vec_env,
		verbose=1,
		learning_rate=0.005,
		clip_range=0.3,
		gae_lambda=0.95,
		gamma=0.99,
		n_steps=2048,
		batch_size=64,
		ent_coef=0.1,
		vf_coef=0.25,
		max_grad_norm=0.3,
		policy_kwargs=dict(
			net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
			activation_fn=torch.nn.LeakyReLU,
		)
	)

	# Configure training callbacks
	print("Setting up callbacks...")
	checkpoint_callback = CheckpointCallback(
		save_freq=MODEL_SAVE_FREQ,
		save_path=MODEL_SAVE_PATH,
		name_prefix='ppo_market_maker'
	)
	training_logging_callback = TrainingLoggingCallback()

	# Train the model
	print("Training the PPO agent...")
	model.learn(
		total_timesteps=TOTAL_TIMESTEPS,
		callback=[checkpoint_callback, training_logging_callback]
	)

	# Save environment normalization statistics
	vec_env.save(os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl"))
	print(f"Saved VecNormalize statistics.")

	# Evaluate on new data
	eval_df = fetch_historical_stock_data(ticker="AAPL", period="8d", interval="1m")
	env_df.head()
	eval_env = DummyVecEnv([lambda: StockMarketMakingEnv(df=eval_df, max_offset=1.0, lot_size=100, inventory_penalty_coeff=0.001, start_index=0, end_index=len(eval_df) - 1)])
	eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

	print("Evaluating on a different time period...")
	avg_reward, avg_pnl = evaluate_model(model, eval_env, num_episodes=5)
	print(f"Evaluation Results - Avg Reward: {avg_reward:.2f}, Avg PnL: {avg_pnl:.2f}")

	# Visualization of agent performance
	print("\nRunning a single episode with visualization:")
	inventory_history = []
	pnl_history = []
	bid_offsets = []
	ask_offsets = []
	agent_bids = []
	agent_asks = []
	market_bids = []
	market_asks = []
	market_mid_prices = []
	market_spreads = []

	obs = eval_env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, infos = eval_env.step(action)
		if done:
			break
		underlying_env = eval_env.envs[0]
		inventory_history.append(underlying_env.inventory)
		pnl_history.append(infos[0].get("pnl", 0.0))
		bid_offsets.append(action[0][0])
		ask_offsets.append(action[0][1])
		row = underlying_env.df.iloc[underlying_env.current_index - 1]
		mid_price = 0.5 * (row["High"] + row["Low"])
		agent_bids.append(mid_price + action[0][0])
		agent_asks.append(mid_price + action[0][1])
		market_bids.append(row["Low"])
		market_asks.append(row["High"])
		market_mid_prices.append(mid_price)
		market_spreads.append(row["High"] - row["Low"])

	plt.figure(figsize=(15, 10))
	plt.subplot(3, 1, 1)
	plt.plot(market_mid_prices, label='Market Mid Price')
	plt.plot(agent_bids, label='Agent Bid')
	plt.plot(agent_asks, label='Agent Ask')
	plt.legend()
	plt.title('Quotes and Market Prices')

	plt.subplot(3, 1, 2)
	plt.plot(inventory_history)
	plt.title('Inventory Over Time')

	plt.subplot(3, 1, 3)
	plt.plot(pnl_history)
	plt.title('PnL Over Time')

	plt.tight_layout()
	plt.show()

# # Run hyperparameter tuning
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)
# print(f"Best hyperparameters: {study.best_params}")
# Visualize optimization results
# fig = optuna.visualization.plot_param_importances(study)
# fig.show()

# fig2 = optuna.visualization.plot_optimization_history(study)
# fig2.show()
