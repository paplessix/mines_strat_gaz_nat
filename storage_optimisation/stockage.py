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
        Y_1 = self.Y_1
        Y_0 = self.Y_0
        Y_2 = self.Y_2
        V = np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)
        def sout_correction( v ):
            stock_level = v/self.Vmax
            if stock_level <Y_1[0]: # A rendre +modualire
                return self.D_nom_sout/(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
            else : 
                return self.D_nom_sout/(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))
        sout_correction  = np.vectorize(sout_correction)
        return -self.Vmax/sout_correction(V)

    def inj_corrige(self,X):
        Y_0 = self. X_0
        Y_1 = self.X_1
        Y_2 = self.X_2    

        V = np.dot(self.m.triang_inf,X) + self.Vinit*np.ones(self.N)
        def inj_correction(v):
            stock_level = v/self.Vmax
            if stock_level < Y_1[0]:# A Rendre + modulaire
                return self.D_nom_inj/(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
            else : 
                return self.D_nom_inj/(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))
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
        self.Y_0 = [0,0.4]
        self.Y_1 = [0.17, 0.65]
        self.Y_2 = [1,1]

        # Capacités d'injection
        self.D_nom_inj  = 85 
        self.X_0 = [0,1]
        self.X_1 = [0.6, 1]
        self.X_2 = [1,0.43]


        self.months_con = {'04' : [0,0.4],'06' :[0.2,0.65],
                            '08': [0.5,0.9],'09' : [0,0.95],
                            '11':[0.85,1]}

class Saline_20 (Stockage):
    # Pas à jour
    def __init__(self,Vmax,Vinit, dates, evolution):
        Stockage.__init__(self, Vmax, Vinit,dates, evolution)
        self.type = 'Saline_20'

        # Capacités de soutirage
        self.D_nom_sout = 17
        self.Y_0 = [0,0.4]
        self.Y_1 = [0.17, 0.65]
        self.Y_2 = [1,1]

        # Capacités d'injection
        self.D_nom_inj  = 85 
        self.X_0 = [0,1]
        self.X_1 = [0.6, 1]
        self.X_2 = [1,0.43]
        
        # Threshold
        self.months_con = {'11':[0.85,1]}

