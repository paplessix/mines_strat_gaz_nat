"""
Module that models a  Gaz - Stockage  with different
paramaters on injection, racking, capacity and 
"""
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matrices import Matrices


class Stockage():
    """
    Class that implements the storage properties
    """
    def __init__(self, Vmax, Vinit, data, evolution, comp_tunnel = True):
        # Base parameters
        self.Vmax = Vmax
        self.Vinit = Vinit
        
        # Time-Series parameters
        self.evolution = evolution
        self.data  = data
        self.dates = data['Day']
        self.index_i = data.index[0]
        self.index_f = data.index[-1]

        # Very useful Parameters
        self.N = len(self.dates)
        self.m = Matrices(self.N)
        self.type = "generique"

        # Capacités de soutirage - ici Générique Sediane Nord 20
        self.D_nom_sout = 44
        Y_0 = [0,0.4]
        Y_1 = [0.17, 0.65]
        Y_2 = [1,1]
        self.sout_list = [Y_0,Y_1,Y_2]

        # Capacités d'injection - ici Générique Sediane nord 20
        self.D_nom_inj  = 85 
        X_0 = [0,1]
        X_1 = [0.6, 1]
        X_2 = [1,0.43]
        self.inj_list = [X_0,X_1,X_2]

        # Contraintes de remplissage - ici Générique  Sediane nord 20
        self.months_con = {'04' : [0,0.4],'06' :[0.2,0.65],
                                    '08': [0.5,0.9],'09' : [0,0.95],
                                    '11':[0.85,1]}

        # calcul du tunnel
        if comp_tunnel : 
            self.tunnel()
    
    def volume_vect (self):
        """ Function that computes the time-serie of the volume
        the function uses a triangular matrix and the initial volume to compute the volume 
        """
        #
        v_vect = self.Vmax*np.dot(self.m.triang_inf,self.evolution) + self.Vinit*np.ones(self.N)
        return v_vect

    v = property(volume_vect)
    
    def volume_calc(self) :
        """
        Compute the final volume in the storage 
        Return : 
            - float
        """
        return self.volume_vect()[-1]
    
    volume_end = property (volume_calc)


    
    def sout_correction(self, v : np.array ):
        """Function that computes the effective maximale racking of the storage 
        parameters :
            - Effective gas volume in the storage
        Return : 
            - Maximal percentage of the Vmax rackable 
        """
        stock_level = v/self.Vmax
        for i in range(1,len(self.sout_list)): 
            if stock_level <= self.sout_list[i][0]: # A rendre +modualir
                return 1/self.D_nom_sout*(self.sout_list[i-1][1] + (self.sout_list[i][1] - self.sout_list[i-1][1])/(self.sout_list[i][0]-self.sout_list[i-1][0])*(stock_level-self.sout_list[i-1][0]))
        return 0

    def sout_corrige(self,X : np.array):
        """
        Function that for an evolution time serie determines the maximal racking
        Parameters : 
            - Effective evolution in the storage 
        Return : 
            - np.array() : Maximal percentage of the Vmax rackable in the storage each day 
        """
        V = self.Vmax*np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)
        sout_correction  = np.vectorize(self.sout_correction)
        return -1*sout_correction(V) 
    
    def inj_correction(self,v):
        """Function that computes the effective maximal injection of the storage 
        parameters :
            - Effective gas volume in the storage
        Return : 
            - Maximal percentage of the Vmax injectable 
        """
        stock_level = v/self.Vmax
        for i in range(1,len(self.inj_list)):
            if stock_level <= self.inj_list[i][0]:# A Rendre + modulaire
                return 1/self.D_nom_inj*(self.inj_list[i-1][1] + (self.inj_list[i][1]-self.inj_list[i-1][1])/(self.inj_list[i][0]-self.inj_list[i-1][0])*(stock_level-self.inj_list[i-1][0]))
        return 0
    def inj_corrige(self,X):
        """
        Function that for an evolution time serie determines the maximal injection each day
        Parameters : 
            - Effective evolution in the storage 
        Return : 
            - np.array() : Maximal percentage of the Vmax injectable in the storage each day 
        """  
        V = self.Vmax*np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)
        inj_correction = np.vectorize(self.inj_correction)
        return inj_correction(V)
    
    ###################### A REFAIRE
    def threshold(self):
        begin = self.dates.min()
        end = self.dates.max()
        begin_month = begin.month
        begin_year = begin.year
        end_month = end.month

        threshold = {}
        for month in self.months_con.keys():
            if datetime.datetime(begin_year,int(month),1) < begin:
                if datetime.datetime(begin_year + 1,int(month),1) < end :
                    index_number = []
                    current_search_day = datetime.datetime(begin_year +1,int(month),1)
                    while len(index_number) == 0 : 
                        index_number =self.data[self.data['Day'] == current_search_day].index.values.astype(int)
                        current_search_day = current_search_day - datetime.timedelta(days = 1)
                    threshold[index_number[0]]= self.months_con[month]
                else :
                    pass
            else:
                if datetime.datetime(begin_year,int(month),1) < end :
                    index_number = []
                    current_search_day = datetime.datetime(begin_year,int(month),1)
                    while len(index_number) == 0 : 
                        index_number =self.data[self.data['Day'] == current_search_day].index.values.astype(int)
                        current_search_day = current_search_day - datetime.timedelta(days = 1)
                    threshold[index_number[0]] = self.months_con[month]
                else:
                    pass

                if datetime.datetime(begin_year+1,int(month),1) < end :
                    index_number = []
                    current_search_day = datetime.datetime(begin_year+1,int(month),1)
                    while len(index_number) == 0 : 
                        index_number =self.data[self.data['Day'] == current_search_day].index.values.astype(int)
                        current_search_day = current_search_day - datetime.timedelta(days = 1)
                    threshold[index_number[0]] = self.months_con[month]
                else:
                    pass
        return threshold

    threshold_con = property(threshold)
    
    def min_v(self):
        """Function that creates the minimum limit of volume 
        The function values 0 everywhere except at the thresholds
        where it values according to the documentation
        Return :
        - Vector of minimum volume each day : np.array
        """
        self.data.loc[:,'minV'] = 0
        for i in self.threshold_con.keys():
            self.data.loc[i,'minV'] = self.threshold_con[i][0]      
        self.data.loc[:,'minV'] = self.Vmax*self.data['minV'] 
        vect_min = np.array(self.data['minV'])
        return vect_min

    def max_v(self):
        """Function that creates the maximum limit of volume
        The function values Vmax everywhere except at the thresholds
        where it values according to the documentation
        Return :
            - Vector of maximum volume each day  : np.array
        """
        self.data.loc[:,"maxV"] = 1
        for i in self.threshold_con.keys():
            self.data.loc[i,'maxV'] = self.threshold_con[i][1]
        self.data.loc[:,'maxV'] = self.Vmax*self.data['maxV']
        vect_max = np.array(self.data['maxV']) 
        return vect_max
    
    vect_min = property (min_v)
    vect_max = property (max_v)
    
    def tunnel_max(self):
        """ Function that computes the maximum value of the volume permitted
        by the constraints

        The function computes the min between the vector returned by Stockage.min_v
        and the vectors returned by Stockage.chapiteau_max. Where Stockage.chapiteau_max returns
        the maximal behaviour authorized by the constraints, meaning the limit behaviour to respect
        the constraint of the beginning of some months.
            e-g : Les positions autorisés en ne faisant que acheter avant et que vendre après le 1 aout,
            tout en étant à 90 % de remplissage ce jour là.
        Parameters:
            - self
        Return :
            - Vector of minimum volume each day  : np.array
        """
        L=np.zeros((len(self.threshold_con.keys()),self.N)) # Construire directement à la bonne taile 
        
        def chapiteau_max(i : int,dirac : float):
            """Function that computes the maximum feasible behaviour, starting symetrically from
            a day of legal obligation 
            Parameters :
                - self
                - index : position of the threshold
                - dirac : value of the threshold

            Return:
                - np.array : which is the limit evolution. 
            """
            serie = np.zeros(self.N)
            i_max = self.N-1
            i_min = 0
            serie[i] = dirac
            for pas in range(1,self.N):
                if i + pas <= i_max: # verify it is still in the limits
                    # inject as much gas as possible moving forward
                    serie[i+pas] = serie[i+pas-1] + self.inj_correction(self.Vmax*serie[i+pas-1])
                if i - pas >= i_min :
                    # rack as much gas as possible moving backward 
                    serie[i-pas] = serie[i-pas+1] + self.sout_correction(self.Vmax*serie[i-pas+1])
            return serie        
        for index, j  in enumerate(self.threshold_con.keys()):
            L [index] = (self.Vmax*chapiteau_max(j -self.index_i,self.threshold_con[j][1]))
        L = np.vstack((L,self.Vmax*np.ones(self.N))) # add a line of Vmax (superior limit)
        return np.array(L).min(axis =0) 

    def tunnel_min(self):
        """Function that computes the minimum value of the volume permitted
        by the constraints

        Symétric from tunnel_max

        Parameters:
            - self
        Return :
            - Vector of minimum volume each day  : np.array
        """
        L=np.zeros((len(self.threshold_con.keys()),self.N))
        def chapiteau_min(i,dirac):
            """Function that computes the minimum feasible behaviour, starting symetrically from
            a day of legal obligation 
            Parameters :
                - self
                - index : position of the threshold
                - dirac : value of the threshold

            Return:
                - np.array : which is the limit evolution. 
            """

            serie = np.zeros(self.N)
            i_max = self.N-1
            i_min = 0
            serie[i] = dirac
            for pas in range(1,self.N):
                if i + pas <= i_max:
                    # rack as much gas as possible moving forward
                    serie[i+pas] = serie[i+pas-1] - self.sout_correction(self.Vmax*serie[i+pas-1])
                if i - pas >= i_min :
                    # inject as much gas as possible moving backward
                    serie[i-pas] = serie[i-pas+1] - self.inj_correction(self.Vmax*serie[i-pas+1])
            return serie        
        for index, j in enumerate(self.threshold_con.keys()):
            L[index] = (self.Vmax*chapiteau_min(j -self.index_i,self.threshold_con[j][0]))
        L = np.array(L)
        L = np.vstack((L,np.zeros(self.N))) # add a line of zeros (inferior limit)
        return np.array(L).max(axis =0 )

    def tunnel(self):
        """ Function that creates the attribute of the optimization problem solution domain.
        Return :
            - None
        """
        self.lim_min =   self.tunnel_min()
        self.lim_max = self.tunnel_max()
   
    def plot_volume(self):
        """function that plots the volume evolution
        """
        plt.plot(self.dates,self.volume_vect())
    
    def plot_threshold(self):
        """ function that plots the legal obligation thresholds
        """
        plt.plot(self.dates,self.vect_min)
        plt.plot(self.dates,self.vect_max)
    
    def plot_tunnel(self):
        """ function that plots the feasible solutions domain 
        """
        plt.plot(self.dates,self.lim_min)
        plt.plot(self.dates,self.lim_max)
    
    def plot_injection(self):
        """function that plots the evolution, and the evolution limits

        The function plots the evolution, using algebric values, the limits induced by
        the injection/racking constraints are 
        """
        var_sup = self.inj_corrige(self.evolution)
        var_inf = self.sout_corrige((self.evolution))
        plt.plot(self.dates,self.evolution, label = 'evol')
        plt.plot(self.dates, var_sup, label = 'inj_max')
        plt.plot(self.dates, var_inf, label = 'sout_max')

    def plot_inj_sout_param(self): 
        pass

    def __str__(self):
        return f"Stockage de Gaz : \n Type : {self.type} \n Volume max = {self.Vmax} \n  Volume initial = {self.Vinit}\n Temps d'évolutions = {self.N} jours \n"


class Sediane_Nord_20 (Stockage):
    """
    Properties of a Sediane_Nord_20 storage
    """
    def __init__(self,Vmax,Vinit, dates, evolution):
        Stockage.__init__(self, Vmax, Vinit,dates, evolution)
        self.type = 'Sediane_Nord_20'
        
        # Capacités de soutirage
        self.D_nom_sout = 44
        Y_0 = [0,0.4]
        Y_1 = [0.17, 0.65]
        Y_2 = [1,1]
        self.sout_list = [Y_0,Y_1,Y_2]
        # Capacités d'injection
        self.D_nom_inj  = 85 
        X_0 = [0,1]
        X_1 = [0.6, 1]
        X_2 = [1,0.43]
        self.inj_list = [X_0,X_1,X_2]

        self.months_con = {'04' : [0,0.4],'06' :[0.2,0.65],
                            '08': [0.5,0.9],'09' : [0,0.95],
                            '11':[0.85,1]}

class Saline_20 (Stockage):
    """
    Properties of a Saline_20 storage
    """
    def __init__(self,Vmax,Vinit, dates, evolution):
        Stockage.__init__(self, Vmax, Vinit,dates, evolution)
        self.type = 'Saline_20'

        # Capacités de soutirage
        self.D_nom_sout = 17
        Y_0 = [0, 0.1]
        Y_1 = [0.15, 0.7]
        Y_2 = [0.45, 1]
        Y_3 = [1, 1]
        self.sout_list = [ Y_0,Y_1,Y_2,Y_3]
        # Capacités d'injection
        self.D_nom_inj  = 125
        X_0 = [0, 1]
        X_1 = [0.45, 0.8]
        X_2 = [0.9, 0.55]
        X_3 = [1, 0.25]
        self.inj_list = [X_0, X_1, X_2, X_3]
        # Threshold
        self.months_con = {'11':[0.85,1]}



class Serene_Atlantique_20 ( Stockage) :
    """
    Properties of a Serene_Atlantique_20 storage
    """

    def __init__(self,Vmax,Vinit, dates, evolution):
        Stockage.__init__(self, Vmax, Vinit,dates, evolution)
        self.type = 'Serene Atlantique 20 '

        # Capacités de soutirage
        self.D_nom_sout = 80
        Y_0 = [0, 0.2]
        Y_1 = [0.15, 0.42]
        Y_2 = [0.55, 0.75]
        Y_3 = [0.8, 0.9]
        Y_4 = [1,1]
        self.sout_list = [ Y_0,Y_1,Y_2,Y_3, Y_4]
        # Capacités d'injection
        self.D_nom_inj  = 135
        X_0 = [0, 1]
        X_1 = [0.3, 1]
        X_2 = [0.65, 0.85]
        X_3 = [0.75, 0.8]
        X_4 = [1, 0.78]
        self.inj_list = [X_0, X_1, X_2, X_3, X_4]
        # Threshold
        self.months_con = {'04':[0,0.4], '08':[0,0.9], '09':[0,0.95], '11':[0.9,1]}


class Serene_Nord_20 ( Stockage) :
    """
    Properties of a Serene_Nord_20 storage
    """
        
    def __init__(self,Vmax,Vinit, dates, evolution):
        Stockage.__init__(self, Vmax, Vinit,dates, evolution)
        self.type = 'Serene Nord 20 '

        # Capacités de soutirage
        self.D_nom_sout = 85
        Y_0 = [0, 0.25]
        Y_1 = [0.1, 0.3]
        Y_2 = [0.35, 0.6]
        Y_3 = [0.7, 0.85]
        Y_4 = [1,1]
        self.sout_list = [ Y_0,Y_1,Y_2,Y_3, Y_4]
        # Capacités d'injection
        self.D_nom_inj  = 135
        X_0 = [0, 1]
        X_1 = [0.3, 1]
        X_2 = [0.65, 0.85]
        X_3 = [0.75, 0.8]
        X_4 = [1, 0.78]
        self.inj_list = [X_0, X_1, X_2, X_3, X_4]
        # Threshold
        self.months_con = {'04':[0,0.4], '08':[0,0.9], '09':[0,0.95], '11':[0.9,1]}

def main():
    """
    Main function that shows some state of the storage
    """
    path_spot = Path(__file__).parent.parent / 'Data' / 'spot_history_HH.csv'

    data =  pd.read_csv(path_spot)

    data = data.iloc[50:450]
    data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')
    X_0 = np.zeros( len(data['Day']))
    
    x = np.linspace(1,99,1000)
    L= []
    for _ in x : 
        stock = Saline_20 (100,100, data, X_0)
        L.append(max(stock.sout_corrige(stock.evolution)*stock.D_nom_sout))
    plt.plot(x,L)
    plt.show()


def main2():
    """
    Main function that shows the tunnel
    """
    path_spot = Path(__file__).parent.parent / 'Data' / 'spot_history_HH.csv'

    data =  pd.read_csv(path_spot)

    data_select = data.iloc[50:500].copy()
    data_select.loc[:,'Day'] = pd.to_datetime(data_select['Day'], format = '%Y-%m-%d')
    X_0 = np.zeros( len(data_select['Day']))
    stock = Stockage (100,10, data_select, X_0)
    stock.tunnel()
    stock.plot_threshold()
    stock.plot_tunnel()
    plt.show()

if __name__ == "__main__":
    sys.exit(main2())

