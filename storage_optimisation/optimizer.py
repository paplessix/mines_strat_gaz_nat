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
        self.c4 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_sout(x), 0,np.inf)
        self.c5 = scipy.optimize.NonlinearConstraint(lambda x : self.con_max_inj(x), 0,np.inf)

    def optimize(self):
        res = minimize(lambda x:profit(x, self.prices), self.X_0 ,method='SLSQP', constraints = [self.c1,self.c2,self.c3,self.c4,self.c5])
        self.stock.evolution = res.x









# #data = pd.read_csv('./storage_optimisation/spot_history_HH.csv')
# data =  pd.read_csv('spot_history_HH.csv')
# data = data.iloc[-300 :-100 ]
# data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')
# plt.plot(data['Day'], data['Price'])
# plt.show()
# N = len(data['Day'])


# months = {'04' : [0,0.4],'06' :[0.2,0.65] ,'08': [0.5,0.9],'09' : [0,0.95],'11':[0.85,1]}
# # Attention il va peut être falloir rajouter un deuxième mois d'avril pour boucler
# X_0 = np.ones(2*N,)
# begin = data['Day'].min()
# end = data['Day'].max()
# begin_month = begin.month
# begin_year = begin.year
# end_month = end.month

# threshold = {}
# for month in months.keys():
#     if datetime.datetime(begin_year,int(month),1) < begin:
#         if datetime.datetime(begin_year + 1,int(month),1) < end :
#             index_number = []
#             current_search_day = datetime.datetime(begin_year +1,int(month),1)
#             while len(index_number) == 0 : 
#                 index_number =data[data['Day'] == current_search_day].index.values.astype(int)
#                 current_search_day = current_search_day - datetime.timedelta(days = 1)
#             threshold[index_number[0]]= months[month]
#         else :
#             pass
#     else:
#         if datetime.datetime(begin_year,int(month),1) < end :
#             index_number = []
#             current_search_day = datetime.datetime(begin_year,int(month),1)
#             while len(index_number) == 0 : 
#                 index_number =data[data['Day'] == current_search_day].index.values.astype(int)
#                 current_search_day = current_search_day - datetime.timedelta(days = 1)
#             threshold[index_number[0]] = months[month]
#         else:
#             pass

#         if datetime.datetime(begin_year+1,int(month)+1,1) < end :
#             index_number = []
#             current_search_day = datetime.datetime(begin_year+1,int(month)+1,1)
#             while len(index_number) == 0 : 
#                 index_number =data[data['Day'] == current_search_day].index.values.astype(int)
#                 current_search_day = current_search_day - datetime.timedelta(days = 1)
#             threshold[index_number[0]] = months[month]
#         else:
#             pass
# data['minV'] = 0
# data["maxV"] = 1
# Vmax = 100
# initial_v = 80

# for i in threshold.keys():
#     data['minV'].loc[i] = threshold[i][0]
#     data['maxV'].loc[i] = threshold[i][1]
# data['minV'] = Vmax*data['minV'] 
# data['maxV'] = Vmax*data['maxV'] 
# plt.plot(data['Day'],data['minV'])
# plt.plot(data['Day'],data['maxV'])

# Vect_min = np.array(data['minV'])
# Vect_max = np.array(data['maxV'])



# def volume(X,initial_v = 0):
#     return initial_v + sum([buy-sell for sell, buy in zip(X[::2],X[1::2])])

# ###Contraints
# I_sell = np.zeros((N,2*N))
# for i in range(0,N):
#     I_sell[i,2*i]=1

# c1 = scipy.optimize.LinearConstraint(I_sell,0,np.inf)

# I_buy = np.zeros((N,2*N))
# for i in range(N):
#     I_buy[i,2*i+1]=1
# c2 = scipy.optimize.LinearConstraint(I_buy,0,np.inf)

# I_diff= np.zeros((N,2*N))

# for i in range(0,N):
#     I_diff[i,2*i]= -1
#     I_diff[i,2*i+1] = +1


# triang_sup = np.zeros((N,N))

# for i in range(0, N) :
#     for j in range(i,N):
#         triang_sup[i,j] = 1

# initial_vect = initial_v * np.ones(N)
# max_vect = Vmax * np.ones(N)
# c3 = scipy.optimize.LinearConstraint(triang_sup.transpose()@I_diff,Vect_min- initial_vect ,Vect_max - initial_vect)
# #c3 = {'type': 'ineq', 'fun': lambda x : Vmax - volume(-x,initial_v)}


# ### Contraintes non linéaires
# def sout_corrige(v,Vmax):
#     Y_0 = [0,0.4]
#     Y_1 = [0.17, 0.65]
#     Y_2 = [1,1]
#     D_nom = 44
#     D_reel = 60
#     # Dans le cas Sediane nord
#     stock_level = v/Vmax
#     if stock_level < Y_1[0]:
#         return D_nom*(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
#     else : 
#         return D_nom*(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))

# sout_corrige  = np.vectorize(sout_corrige)

# def inj_corrige(v,Vmax):
#     Y_0 = [0,1]
#     Y_1 = [0.6, 1]
#     Y_2 = [1,0.43]
#     D_nom = 85
#     D_reel = 101
#     # Dans le cas Sediane nord
#     stock_level = v/Vmax
#     if stock_level < Y_1[0]:
#         return D_nom*(Y_0[1] + (Y_1[1]-Y_0[1])/(Y_1[0]-Y_0[0])*stock_level)
#     else : 
#         return D_nom*(Y_1[1] + (Y_2[1]-Y_1[1])/(Y_2[0]-Y_1[0])*(stock_level-Y_1[0]))

# inj_corrige = np.vectorize( inj_corrige)

# def max_soutirage(x):
#     v_vect = np.dot(triang_sup.transpose()@I_diff,x) + initial_vect
#     max_x = sout_corrige(v_vect, Vmax)
#     return Vmax/max_x

# def max_injection(x):
#     v_vect = np.dot(triang_sup.transpose()@I_diff,x) + initial_vect
#     max_x = inj_corrige(v_vect, Vmax)
#     return Vmax/max_x

# def con_max_sout(x):
#     return max_soutirage(x)- np.dot(I_sell,x)

# def con_max_inj(x):
#     return max_injection(x)-np.dot(I_buy,x)

# c4 = scipy.optimize.NonlinearConstraint(con_max_sout,0,np.inf)
# c5 = scipy.optimize.NonlinearConstraint(con_max_inj,0,np.inf)
# res = minimize(lambda x:profit(x, data['Price']), X_0,method='SLSQP', constraints = [c1,c2,c3,c4,c5])
# print(-profit(res.x,data['Price']))


# volumes=[]
# for i in range(2,2*(N+1),2):
#     volumes.append(volume(res.x[:i], initial_v))


# plt.plot(data['Day'],volumes)

# plt.show()
