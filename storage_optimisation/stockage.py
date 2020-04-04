import matplotlib.pyplot as plt 
import numpy as np
from matrices import Matrices

class Stockage():
    def __init__(self, Vmax, Vinit, dates, evolution):
        self.Vmax = Vmax
        self.Vinit = Vinit
        self.evolution = evolution
        self.dates = dates
        self.N = len(dates)
        self.m = Matrices(self.N)

        #Propriétés du stockage- version générique 
        self.Y_0 = [0,0.4]
        self.Y_1 = [0.17, 0.65]
        self.Y_2 = [1,1]
        self.D_nom = 44
    
    def volume_vect (self):
        v_vect = np.dot(self.m.triang_inf@self.m.I_diff,self.evolution) + self.Vinit*np.ones(self.N)
        return v_vect
    v = property(volume_vect)
    def volume_calc(self) :
        return self.volume_vect()[-1]
    
    volume_end = property (volume_calc)

    def plot_volume(self):
        plt.plot(self.dates,self.volume_vect())
        plt.show()
    
    def sout_corrige(self):
        Y_1 = self.Y_1
        Y_0 = self.Y_0
        Y_2 = self.Y_2
        def sout_correction( v ):
            stock_level = v/self.Vmax
            if stock_level < self.Y_1[0]: # A rendre +modualire
                return self.D_nom*(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
            else : 
                return self.D_nom*(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))
        sout_correction  = np.vectorize(sout_correction)
        return self.Vmax/sout_correction(self.v)

    def inj_corrige(self):
        Y_1 = self.Y_1
        Y_0 = self.Y_0
        Y_2 = self.Y_2
        def inj_correction(v):
            stock_level = v/self.Vmax
            if stock_level < Y_1[0]:# A Rendre + modulaire
                return self.D_nom*(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
            else : 
                return self.D_nom*(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))
        inj_correction = np.vectorize(inj_correction)
        return self.Vmax/inj_correction(self.v)

        

    
class Sediane_Nord_20 (Stockage):
    def __init__(self,Vmax,Vinit):
        Stockage.__init__(self, Vmax, Vinit, dates, evolution )
        self.type = 'Sediane_Nord_20'
        self.Y_0 = [0,0.4]
        self.Y_1 = [0.17, 0.65]
        self.Y_2 = [1,1]
        self.D_nom = 44

stock = Stockage(100, 50, np.array([0,1,2,3,4,5,6,7,8,9]), np.ones((20,)))
print('Volume',stock.volume_end)

stock.plot_volume()