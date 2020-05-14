from optimizer import Optimizer, profit
from stockage import Stockage
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Simulation : 
    def __init__ (self):
        pass
    def data_loader_csv(self, path):
        df = pd.read_csv(path, header = 0, index_col = 0)
        df  =df .transpose()
        self.data = df.iloc[:]
        self.data.index = pd.to_datetime(self.data.index)
        self.columns = self.data.columns
        self.X_0 = np.zeros(len(self.data.index))
        
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
        for column in self.columns[0 :80 ]:
            print('Optimizing:', column )
            df_inter["Price"] = self.data[column].values
            stock = Stockage(100, V_0, df_inter, self.X_0)
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
        plt.show()
    
    def value_at_risk(self, disp = True, verbose = True  ):
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
    