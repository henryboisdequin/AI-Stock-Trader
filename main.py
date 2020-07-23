import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from datetime import datetime
import itertools
import argparse
import os
import pickle
from sklearn.preprocessing import StandardScaler


def get_data():
    """
    Gets data from CSV (0 - AAPL, 1 - MSI, 2 - SBUX).
    :return: T x 3 Stock Prices
    """
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values


class ReplayBuffer:
    """
    Experience replay memory.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.act_dim = act_dim
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        Stores state, action reward in respected buffers.
        :return: None
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size_=32):
        """
        Chooses random indices for the buffer.
        :return: dict
        """
        idxs = np.random.randint(0, self.size, size=batch_size_)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


def get_scaler(env_):
    """
    :return: scaler object
    """

    states = []
    for _ in range(env_.n_step):
        action = np.random.choice(env_.action_space)
        state, reward, done, info = env_.step(action)
        states.append(state)
        if done:
            break

    scaler_ = StandardScaler()
    scaler_.fit(states)
    return scaler_


def make_dir(directory):
    """
    Function to make directory if needed.
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
    """
    Makes a MLP neural network model.
    :return: model object
    """

    # Input layer
    i = Input(shape=(input_dim,))
    x = i

    # Hidden layers
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)

    # Dense layer
    x = Dense(n_action)(x)

    # Create the model
    model = Model(i, x)

    model.compile(loss='mse', optimizer='adam')
    print((model.summary()))  # summarizes model
    return model


class StockTradingEnv:
    """
    A 3-stock trading environment for our AI agent.
    0 - sell
    1 - hold
    2 - buy
    """

    def __init__(self, data_, initial_investment_=20000):
        self.stock_price_history = data_
        self.n_step, self.n_stock = self.stock_price_history.shape
        self.initial_investment = initial_investment_
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.action_space = np.arange(3 ** self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2],  # 27 possible
                                                            # actions
                                                            repeat=self.n_stock)))
        self.state_dim = self.n_stock * 2 + 1
        self.reset()

    def reset(self):
        """
        Resets to initial investment and resets all stock trading history.
        :return: state vector
        """
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        """
        Performs action in environment.
        :return: state vector, reward, done, info
        """
        assert action in self.action_space

        prev_val = self._get_val()

        # Update price for each day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # Perform the trade
        self._trade(action)

        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        Returns vector of three components.
        :return: state
        """
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        """
        Gets value we have in our portfolio.
        :return:
        """
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        """
        Determines whether to sell/buy/hold.
        :return: None
        """
        action_vec = self.action_list[action]

        # Determine which stocks to buy or sell
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0

        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # Buying shares
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


class DQNAgent(object):
    """
    This is our artificial intelligence agent, the "decision maker" of stock trading
    in our environment.
    """

    def __init__(self, state_size_, action_size_):
        self.state_size = state_size_  # Inputs of NN (Neural Network)
        self.action_size = action_size_  # Outputs of NN
        self.memory = ReplayBuffer(state_size_, action_size_, size=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size_, action_size_)

    def update_replay_memory(self, state, action, reward, next_state, done):
        """
        Stores everything the DQN Agent needs in memory.
        :return:
        """
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        """
        Uses epsilon greedy to choose an action based on the state parameter.
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def replay(self, batch_size_=32):
        """
        Where the AI learns from its mistakes and victories.
        :return: None
        """
        if self.memory.size < batch_size_:
            return

        mini_batch = self.memory.sample_batch(batch_size_)
        states = mini_batch['s']
        actions = mini_batch['a']
        rewards = mini_batch['r']
        next_states = mini_batch['s2']
        done = mini_batch['d']

        # Calculate the target
        target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)

        target_full = self.model.predict(states)
        target_full[np.arange(batch_size_), actions] = target

        # Run one training step
        self.model.train_on_batch(states, target_full)  # gradient descent

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Load model weights.
        :return: None
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Save model weights.
        :return: None
        """
        self.model.save_weights(name)


def play_one_episode(agent_, env_, is_train):
    """
    Plays an episode of stock trading.
    :return:
    """
    state = env_.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent_.act(state)
        next_state, reward, done, info = env_.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent_.update_replay_memory(state, action, reward, next_state, done)
            agent_.replay(batch_size)
        state = next_state

    return info['cur_val']


if __name__ == '__main__':
    """
    Main code for the agent, environment, and stock trading actions.
    """
    # Config
    models_folder = 'rl_trader_models'  # stores models
    rewards_folder = 'rl_trader_rewards'  # stores rewards
    num_episodes = 20
    batch_size = 32
    initial_investment = 20000
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    make_dir(models_folder)
    make_dir(rewards_folder)

    # Get time series
    data = get_data()
    n_time_steps, n_stocks = data.shape

    n_train = n_time_steps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    # Create environment with data
    env = StockTradingEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # Final value of portfolio
    portfolio_value = []

    if args.mode == 'test':  # test mode
        # Load previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Remake environment
        env = StockTradingEnv(test_data, initial_investment)

        agent.epsilon = 0.01

        # Load trained weights
        agent.load(f'{models_folder}/dqn.h5')

    # Play the 'game'
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"Episode: {e + 1}/{num_episodes}, Episode end Value: {val:.2f}, Duration: {dt}")
        portfolio_value.append(val)  # Append episode end portfolio value to track progress

    # Save weights when done with episode
    if args.mode == 'train':  # train mode 
        # Save the DQN agent
        agent.save(f'{models_folder}/dqn.h5')

        # Save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # Save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
