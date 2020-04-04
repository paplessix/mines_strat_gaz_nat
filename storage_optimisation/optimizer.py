import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy

data = pd.read_csv('spot_history_HH.csv')
data = data.iloc[-50:]
#data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')
plt.plot(data['Day'], data['Price'])
plt.show()
N = len(data['Day'])

def profit(X,prices) : 
    return  -sum([price*sell - buy*price for sell,buy,price in zip(X[::2],X[1::2],prices)])

X_0 = np.ones(2*N,)

prof  = profit(X_0 , data['Price'])
print( prof, sum(data['Price']))



Vmax = 100

def volume(X,initial_v = 0):
    return initial_v + sum([buy-sell for sell, buy in zip(X[::2],X[1::2])])

###Constraits
I_sell = np.zeros((2*N,2*N))
for i in range(0,2*N,2):
    I_sell[i,i]=1

c1 = scipy.optimize.LinearConstraint(I_sell,0,0.016*Vmax)

I_buy = np.zeros((2*N,2*N))
for i in range(1,2*N+1,2):
    I_buy[i,i]=1
c2 = scipy.optimize.LinearConstraint(I_buy,0,0.016*Vmax)

I_diff= np.zeros((2*N,2*N))
for i in range(0,2*N,2):
    I_diff[i,i]= -1
    I_diff[i,i+1] = +1

triang_sup = np.zeros((2*N,2*N))
for i in range(0, 2*N):
    for j in range(i,2*N):
        triang_sup[i,j] = 1

c3 = scipy.optimize.LinearConstraint(triang_sup@I_diff,0,Vmax)
#c3 = {'type': 'ineq', 'fun': lambda x : Vmax - volume(-x)}

res = minimize(lambda x:profit(x, data['Price']), X_0,method='SLSQP', constraints = [c1,c2,c3],options={'disp': True})

volumes=[]
for i in range(0,2*N,2):
    volumes.append(volume(-res.x[:i]))
plt.plot(volumes)
