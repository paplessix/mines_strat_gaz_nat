import matplotlib.pyplot as plt 
import numpy as np
from matrices import Matrices
import datetime
import pandas as pd

class Stockage():
    def __init__(self, Vmax, Vinit, data, evolution):
        self.Vmax = Vmax
        self.Vinit = Vinit
        self.evolution = evolution
        self.data  = data
        self.dates = data['Day']
        self.N = len(self.dates)
        self.m = Matrices(self.N)
        self.type = "generique"
        #Propriétés du stockage- version générique 
        self.Y_0 = [0,0.4]
        self.Y_1 = [0.17, 0.65]
        self.Y_2 = [1,1]
            # Capacités d'injection 
        self.X_0 = [0,1]
        self.X_1 = [0.6, 1]
        self.X_2 = [1,0.43]
        self.D_nom_sout = 44
        self.D_nom_inj  = 85
        self.months_con = {'04' : [0,0.4],'06' :[0.2,0.65],
                                    '08': [0.5,0.9],'09' : [0,0.95],
                                    '11':[0.85,1]}
    
    def volume_vect (self):
        v_vect = np.dot(self.m.triang_inf,self.evolution) + self.Vinit*np.ones(self.N)
        return v_vect

    v = property(volume_vect)
    
    def volume_calc(self) :
        return self.volume_vect()[-1]
    
    volume_end = property (volume_calc)

    def plot_volume(self):
        plt.plot(self.dates,self.volume_vect())
    
    def sout_corrige(self,X):
        sout_list = self.sout_list 
        V = np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)
        def sout_correction( v ):
            stock_level = v/self.Vmax
            for i in range(1,len(sout_list)): 
                if stock_level <sout_list[i][0]: # A rendre +modualire
                    return self.D_nom_sout/(sout_list[i-1][1] + (sout_list[i][1]-sout_list[i-1][1])/(sout_list[i][0]-sout_list[i-1][0])*stock_level)
        sout_correction  = np.vectorize(sout_correction)
        return -self.Vmax/sout_correction(V)

    def inj_corrige(self,X):
        inj_list = self.inj_list  
        V = np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)

        def inj_correction(v):
            stock_level = v/self.Vmax
            for i in range(1,len(sout_list)):
                if stock_level < inj_list[i][0]:# A Rendre + modulaire
                    return self.D_nom_inj/(inj_list[i-1][1] + (inj_list[i][1]-inj_list[i-1][1])/(inj_list[i][0]-inj_list[i-1][0])*stock_level)
        inj_correction = np.vectorize(inj_correction)
        return self.Vmax/inj_correction(V)
    
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
        self.data['minV'] = 0
        for i in self.threshold_con.keys():
            self.data['minV'].loc[i] = self.threshold_con[i][0]      
        self.data['minV'] = self.Vmax*self.data['minV'] 
        Vect_min = np.array(self.data['minV'])
        return Vect_min

    def max_v(self):
        self.data["maxV"] = 1
        for i in self.threshold_con.keys():
            self.data['maxV'].loc[i] = self.threshold_con[i][1]
        self.data['maxV'] = self.Vmax*self.data['maxV']
        Vect_max = np.array(self.data['maxV']) 
        return Vect_max
    
    vect_min = property (min_v)
    vect_max = property (max_v)

    def plot_threshold(self):
        plt.plot(self.dates,self.vect_min)
        plt.plot(self.dates,self.vect_max)
    
    def plot_injection(self):
        var_sup = self.inj_corrige(self.evolution)
        var_inf = self.sout_corrige((self.evolution))
        plt.plot(self.dates, self.evolution, label = 'evol')
        plt.plot(self.dates, var_sup, label = 'inj_max')
        plt.plot(self.dates, var_inf, label = 'sout_max')

    def __str__(self):
        return f"Stockage de Gaz : \n Type : {self.type} \n Volume max = {self.Vmax} \n  Volume initial = {self.Vinit}\n Temps d'évolutions = {self.N} jours \n"


class Sediane_Nord_20 (Stockage):
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
        


