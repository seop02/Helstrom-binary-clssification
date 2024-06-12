# QUEUCO QUANTATIVE RESEARCHER TECHNICAL TASK

This code demonstrates trading strategies I have developed for the technical task for a quantative researcher role.

This folder consists of different parts: `data_preprocessing.ipynb`, `simulator`, `lr_trade.py` and `take.py`.

`logs` and `figures` folder contains log files, and figures generated from the simulation of different trading strategies.

`data_preprocessing.ipynb` illustrates by initial approach in analyzing the given orderbook data. It also demonstrates motivation behind my approach.

`simulator` folder contains necessary python codes that perform simulation of trading strategies. It takes the `lr_trade.py` or `take.py` as an input strategies. 

## HOW TO RUN SIMULATION

To run the simulation, first run `export PYTHONPATH=$(pwd)` on the terminal.
This makes sure that the current directory is in the list of directories that Python will search for modules and packages.

Then, you can run the simulation by running following code.

```
python simulator "path to the trader file" "date" vis
```

For example, if you would like to simulate `lr_trade.py` on `20210521`, you should run

```
python simulator lr_trade.py 20210521 vis
```

The `vis` command at the end ensures that the plot of `balance`, `profit`, and `transactions` are generated at the end of the simulations. These figures would be stored at the folder `figures` automatically.

If you do not want such visualizations to be generated you can simply type any string other than `vis`.

## UTILIZING THIS PIPELINE

We can utilize this pipeline to test various trading strategies in real trading environment by simply adding new `trader` code. As long as the new trader code is in the format:

```python
class Trader:

    def run(self, state: TradingState, log: logger) -> Tuple[Dict[str, List[Order]], int, str]:

        #code containing trading strategies
        return orders, trader_data, state, log
```

it can always be simulated efficiently.

## MODIFYING TRADING ENVIRONMENT

The user can always make adjustments to the trading environment such as latency, transaction cost, and position limits.

To modify the trading latency, the user can change the `latency` variable in the line 15 in `simulator/run.py`. 

To change the transaction cost, the user can change the `transaction_cost` variable in the line 23 in `simulator/lr_trade.py` or `simulator/take.py`.

To change the position limit on each asset, the user can change the `LIMIT` variable in the line 13 in `simulator/run.py` AND the `POSITION_LIMIT` variable in the line 30 in `simulator/lr_trade.py` or `simulator/take.py`.
