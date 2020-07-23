# AI-Stock-Trader
Created an DQN AI Agent which chooses to sell, buy, or keep stocks from various companies. This AI agent was made with Tensorflow's Keras API, Pandas, Numpy, and Sklearn. This Agent chooses whether to buy/sell/keep stocks from Apple, Starbucks, and Motorola Solutions. The agent was able to achieve a profit of $20916.63 in 20 episodes of trading stocks. By Henry Boisdequin

<img src="test.png">
Histogram: This image shows the value of the Agents portfolio as the number of episodes increased.

Needed Modules:
```
pip install tensorflow
pip install pandas
pip install numpy
pip install sklearn
```

To Run:
```
To Train: python main.py -m train && python plot_rewards.py -m train
To Test: python main.py -m test && python plot_rewards.py -m test
```

Dataset:
```
aapl_msi_sbux.csv
```
