import os 
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strat_gaz.storage_optimisation.optimizer import Optimizer, profit
from strat_gaz.storage_optimisation.stockage import Stockage

def filename(stockage, size, v_init, index):
    return str(stockage) + '_' + str(size) + '_' + str(v_init) + '__' + str(index) + '.csv'

#### My simulations parameters
INDEX = 'PA'
SIZE = 100
V_INIT = 20
STOCKAGE = None
INPUT_PATH = Path(__file__).parent.parent / 'Data' / 'Diffusion_model.csv'
OUTPUT_PATH = Path(__file__).parent/  filename(STOCKAGE, SIZE, V_INIT, INDEX)

def list_csv_dir(directory : str):
    """
    Sparse all the csv files present in the active directory
    Parameters :
        - directory : str, the working directory
    Return :
        - a list of all the csv files in the directory
    """
    files = list(filter(lambda x: x[-4:] == '.csv', os.listdir(directory)))
    return files

class Simulation:
    def __init__(self, INPUT, OUTPUT,INDEX, SIZE, V_INIT, STOCKAGE):
        self.output = OUTPUT
        self.index = INDEX
        self.size = SIZE
        self.v_init = V_INIT
        self.filename = filename(STOCKAGE, SIZE, V_INIT, INDEX)
        self._fields = ['simulation','profit','volume']
        self.data_loader_csv(INPUT)
        self.output_initializer(self.output)
        
    def data_loader_csv(self, path):
        df = pd.read_csv(path, header = 0, index_col = 0)
        df  =df .transpose()
        self.data = df.iloc[0:50,:]
        self.data.index = pd.to_datetime(self.data.index)
        self.columns = self.data.columns
        self.X_0 = np.zeros(len(self.data.index))

    def output_initializer(self,path):
        list_csv  = list_csv_dir(str(self.output.parent))
        print(list_csv)
        if self.filename in list_csv:
            existing_data = pd.read_csv(OUTPUT_PATH)
            if set(existing_data.columns) != set(self._fields):
                raise TypeError
            self.done = list(existing_data['simulation'])
        else :
            self.done  = []
            header = pd.DataFrame(dict(zip(self._fields, [[], [], []])))
            header.to_csv(self.output, index = False)
    

    def execute(self, number):
        N = len(self.data.columns)
        to_do = [x for x in self.data.columns if x not in self.done]
        for _ in range(number):
            try :
                column = to_do.pop(0)
            except IndexError as e :
                break
            profit, vol_fin = self.optimize(column)
            self.add_line_to_csv(column, profit, vol_fin)
            self.done.append(column)
            print('Writing Done', column)
        print("End of optimization batch. ")


    def optimize(self,column_index):
        df_inter = pd.DataFrame()
        df_inter["Day"] = self.data.index
        df_inter["Price"] = self.data[column_index].values
        
        # Optimization
        print('=================')
        print('Optimizing:', column_index)
        stock = Stockage(self.size, self.v_init, df_inter, self.X_0)
        opti = Optimizer(stock) 
        opti.contraints_init()
        opti.optimize()
        print( 'Optimizing:', column_index, ' Status  = Done' )


        self.X_0 = stock.evolution

        print('initial_investment', self.v_init*df_inter["Price"][0])
        profits =  -self.v_init*df_inter["Price"][0] -profit(stock.Vmax*stock.evolution, df_inter["Price"])
        vol_fin = stock.volume_end
        return profits, vol_fin

    
    def add_line_to_csv(self, simulation_index, profit, vol_fin ):
        for i in zip(self._fields, [[simulation_index], [profit], [vol_fin]]):
            print(i)
        myDict = dict(zip(self._fields, [[simulation_index], [profit], [vol_fin]]))
        print(myDict)
        df = pd.DataFrame(myDict)
        df.to_csv(self.output, mode = 'a', index = False, header = False)


    def plot_data_boxplot(self, mean  = False, Boxplot  = False):
        self.data.T.boxplot()
        plt.show()

simul = Simulation(INPUT_PATH, OUTPUT_PATH, INDEX, SIZE,V_INIT,STOCKAGE)
# simul = plot_data_boxplot()   
simul.execute(50)
