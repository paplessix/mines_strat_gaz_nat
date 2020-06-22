"""
Module that assures the optimization of the evolution attribute of a Gaz Storage
represented by a Stockage object
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy
import datetime
from matrices import Matrices

##### Fonction de Cout #####
def profit(X,prices) : 
        return sum([vol * price for vol,price in zip(X,prices)])

class Optimizer :
    """ Class that handles the optimization of the storage utilisation
    """

    def __init__(self,stockage : "Stockage"):
        """ Constructor
        
        Parameters :
            - stockage : Stockage, the Storage to optimize
        Return:
            - None
        """
        self.prices = stockage.data['Price']
        self.stock = stockage
        self.m = stockage.m
        self.initial_vect  = self.stock.Vinit * np.ones((self.stock.N))
        self.X_0 = np.zeros(self.stock.N,)
    
       
    def con_max_sout(self,x):
        """constraint that limits the racking
        """
        return  - ( self.stock.sout_corrige(x)- x)

    def con_max_inj(self,x):
        """constraint that limits the injection
        """
        return self.stock.inj_corrige(x) - x

    def con_inf(self,x):
        """constraint that assures that the volume isn't too low
        """
        V = self.stock.Vmax*np.dot(self.stock.m.triang_inf,x) + self.stock.Vinit*np.ones(self.stock.N)
        return  V - self.stock.lim_min

    def con_sup(self,x):
        """constraint that assures that the volume isn't too high
        """
        V = self.stock.Vmax*np.dot(self.stock.m.triang_inf,x) + self.stock.Vinit*np.ones(self.stock.N)
        return  self.stock.lim_max - V

    def contraints_init(self):
        """
        Initialization of all the constraints
        """
        # The evolution belongs to [-1,1]
        self.c1 = scipy.optimize.LinearConstraint(np.eye(self.stock.N),-1,1)
        self.c3 = scipy.optimize.LinearConstraint(self.stock.Vmax*self.m.triang_inf,
                                                 self.stock.vect_min - self.initial_vect, 
                                                 self.stock.vect_max - self.initial_vect)

        # Injection / racking constraints
        self.c4 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_sout(x),0, np.inf)
        self.c5 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_inj(x),0, np.inf)
        # Volume constraints
        self.c6 = scipy.optimize.NonlinearConstraint(lambda x : self.con_inf(x),0, np.inf)
        self.c7 = scipy.optimize.NonlinearConstraint(lambda x : self.con_sup(x),0, np.inf)

    def optimize(self):
        """Function that minimize stock.evolution
        
        The function launchs the minimization under constraints using SLSQP (the fastest method)
        the parameters have been set in order to reduce the computing time. We agreed
        to loss useless precision to gain rapidity

        Return :
            - None
        """
        res = minimize(lambda x:profit(self.stock.Vmax*x, self.prices), self.X_0 ,method='SLSQP', constraints = [self.c1,self.c3, self.c4, self.c5, self.c6, self.c7], options={'ftol':1 , 'disp' : True})
        self.stock.evolution = res.x
#self.c1,self.c4,self.c5, self.c6, self.c7