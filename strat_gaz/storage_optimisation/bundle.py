from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from storage_optimisation.optimizer import Optimizer, profit
from storage_optimisation.stockage import Stockage

#### Simulations Parameters
INDEX = 1
SIZE = 10000
V_INIT = 100
STOCKAGE = None
INPUT_PATH = Path(__file__).parent.parent / 'Data' / 'Diffusion_model.csv'
OUTPUT_PATH = Path(__file__).parent /'Output' /  filename(STOCKAGE, SIZE, V_INIT, INDEX)


def filename(stockage, size, v_init, index):
    return str(stockage) + '_' + str(size) + '_' + str(V) + '__' + str(index)

def list_csv_dir(directory : str):
    """
    Sparse all the csv files present in the active directory
    Parameters :
        - directory : str, the working directory
    Return :
        - a list of all the csv files in the directory
    """
    files = list(map(lambda x: directory + '/' + x,
                     filter(lambda x: x[-4:] == '.csv', os.listdir(directory))))
    return files

class Simulation : 
    def __init__ (self  ):
        pass
    def data_loader_csv(self, path):
        df = pd.read_csv(path, header = 0, index_col = 0)
        df  =df .transpose()
        self.data = df.iloc[0:50,:]
        self.data.index = pd.to_datetime(self.data.index)
        self.columns = self.data.columns
        self.X_0 = np.zeros(len(self.data.index))
    
    def add_line_to_csv(simulation, )
        pass

    
    def simulation_plot(self, mean  = False, Boxplot  = False):
        print(self.data.index)
        self.data.T.boxplot()
        #plt.plot(self.data.index,self.data)
        plt.show()
   
    def optimizer(self, V_0 ):
        self.profits = []
        self.strategies = []
        df_inter = pd.DataFrame()
        df_inter["Day"] = self.data.index
        print(df_inter)
        threshold_disp = False
        for column in self.columns[:100]:
            print('=================')
            print('Optimizing:', column )
            df_inter["Price"] = self.data[column].values
            stock = Stockage(10000, V_0, df_inter, self.X_0)
            opti = Optimizer(stock) 
            opti.contraints_init()
            opti.optimize()
            print( 'Optimizing:', column, 'Done' )
            if not threshold_disp:
                threshold_disp = True
                stock.plot_threshold()
            
            stock.plot_volume()
            print('initial_invest', V_0*df_inter["Price"][0])
            self.profits.append( -V_0*df_inter["Price"][0] -profit(stock.evolution, df_inter["Price"]))
            self.strategies.append(list(stock.volume_vect()))
            self.X_0 = stock.evolution
            self.add_line( )
        plt.show()
    
    def value_at_risk(self, disp = True, verbose = True ):
        try:
            profits = self.profits
            print('Nb simulations : ', len(profits))
            print(profits)
            strategies = self.strategies

            Var = np.percentile(profits,10)
            print(Var)
        except AttributeError as e :
            print(e, " Run Simulation.optimizer before")
        if verbose :
            pass
        if disp :
            # graphical display
            fig, (ax1, ax2) = plt.subplots(ncols = 2)

            for i in range (len(strategies)):
                ax1.plot(self.data.index,strategies[i])
            ax2.hist(profits, bins = 20,orientation = 'horizontal')
            ax2.plot([*ax2.get_xlim()],[Var,Var], label = 'Value at Risk')
            ax2.legend()
            plt.show()

simul = Simulation()
path_csv = Path(__file__).parent.parent / 'Data' / 'Diffusion_model.csv'
simul.data_loader_csv(path_csv)
print(simul.data)
simul.simulation_plot()
simul.optimizer(30)
simul.value_at_risk()
