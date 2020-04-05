import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy
import datetime
from matrices import Matrices


def profit(X,prices) : 
        return - sum([price*sell - buy*price for sell,buy,price in zip(X[::2],X[1::2],prices)])

class Optimizer :
    def __init__(self,stockage):
        self.prices = stockage.data['Price']
        self.stock = stockage
        self.m = stockage.m
        self.initial_vect  = self.stock.Vinit * np.ones((self.stock.N))
        self.X_0 = np.ones(2*self.stock.N,)
    
       
    def con_max_sout(self,x):
        return self.stock.sout_corrige(x)- np.dot(self.m.I_sell,x)

    def con_max_inj(self,x):
        return self.stock.inj_corrige(x)-np.dot(self.m.I_buy,x)

    def contraints_init(self):
        #Positivité des échanges
        
        self.c1 = scipy.optimize.LinearConstraint(self.m.I_sell,0,np.inf)
        self.c2 = scipy.optimize.LinearConstraint(self.m.I_buy,0,np.inf)
        self.c3 = scipy.optimize.LinearConstraint(self.m.triang_inf@self.m.I_diff,
                                                self.stock.vect_min - self.initial_vect, 
                                                self.stock.vect_max - self.initial_vect)
        self.c4 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_sout(x),0, np.inf)
        self.c5 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_inj(x),0, np.inf)

    def optimize(self):
        res = minimize(lambda x:profit(x, self.prices), self.X_0 ,method='SLSQP', constraints = [self.c1,self.c2,self.c3,self.c4,self.c5])
        self.stock.evolution = res.x
