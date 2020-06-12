import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import time

from strat_gaz.storage_optimisation.optimizer import Optimizer, profit
from strat_gaz.storage_optimisation.stockage import Stockage


path_spot = Path(__file__).parent.parent / 'Data' / 'spot_history_HH.csv'

data =  pd.read_csv(path_spot)
Time = []
ara = np.arange(10,100,20)
for i in ara :
    print( i )
    a = time()
    datat = data.iloc[50:50+i]
    datat['Day'] = pd.to_datetime(datat['Day'], format = '%Y-%m-%d')
    X_0 = np.zeros( len(datat['Day']))
    stock = Stockage(100,20 , datat, X_0)
    opti = Optimizer(stock) 
    opti.contraints_init()
    opti.optimize()
    b = time()
    Time.append(b-a)
plt.plot(ara,Time)
plt.xlabel("Taille en nombre de jours")
plt.ylabel("Temps de calcul en seconde")
plt.show()  
