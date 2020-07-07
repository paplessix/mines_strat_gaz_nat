from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strat_gaz.storage_optimisation.optimizer import Optimizer
from strat_gaz.storage_optimisation.stockage import Serene_Nord_20, Stockage

plt.close()
path_spot = Path(__file__).parent.parent / 'Data' / 'spot_history_HH.csv'

data = pd.read_csv(path_spot)

data = data.iloc[50:350]

data['Day'] = pd.to_datetime(data['Day'], format='%Y-%m-%d')
#data['Price'] = np.sin(np.linspace(0,6,len(data['Day'])))+1
plt.plot(data['Day'], data['Price'])
plt.show()
plt.close()
X_0 = np.zeros(len(data['Day']))


stock = Stockage(100, 20, data, X_0)
stock.plot_threshold()
stock.tunnel()
stock.plot_tunnel()
stock.plot_volume()
plt.show()
stock.plot_injection()
plt.legend()
plt.show()

print('Volume', stock.volume_end)
print('Threshold', stock.threshold_con)

opti = Optimizer(stock)
opti.contraints_init()
opti.optimize()
stock.plot_threshold()
stock.plot_tunnel()
stock.plot_volume()
plt.show()

plt.close()
stock.plot_injection()
plt.legend()
plt.show()
