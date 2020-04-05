from optimizer import Optimizer
from stockage import Stockage
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data =  pd.read_csv('spot_history_HH.csv')
data = data.iloc[50 :150]
data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')
plt.plot(data['Day'], data['Price'])
plt.show()
X_0 = np.zeros(2*len(data['Day']))


stock = Stockage(100, 50, data, X_0)
#stock.plot_threshold()

print('Volume',stock.volume_end)
print('Threshold', stock.threshold_con)

opti = Optimizer(stock)
opti.contraints_init()
opti.optimize()
stock.evolution
stock.plot_threshold()
stock.plot_volume()
plt.show()
stock.plot_injection()
stock.plot_soutirage()
plt.legend()
plt.show()