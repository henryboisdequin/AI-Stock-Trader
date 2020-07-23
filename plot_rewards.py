import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'rl_trader_rewards/{args.mode}.npy')

print(f"Average reward: {a.mean():.2f}, Min: {a.min():.2f}, Max: {a.max():.2f}")

plt.hist(a, bins=20)
plt.title(args.mode)
plt.show()

"""
To train: ```python main.py -m train && python plot_rewards.py -m train```
To Test: ```python main.py -m test && python plot_rewards.py -m test```
"""
