import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy
import datetime

#data = pd.read_csv('./storage_optimisation/spot_history_HH.csv')
data =  pd.read_csv('spot_history_HH.csv')
data = data.iloc[50 :350 ]
data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')
plt.plot(data['Day'], data['Price'])
plt.show()
N = len(data['Day'])

def profit(X,prices) : 
    return  -sum([price*sell - buy*price for sell,buy,price in zip(X[::2],X[1::2],prices)])

months = {'04' : [0,0.4],'06' :[0.2,0.65] ,'08': [0.5,0.9],'09' : [0,0.95],'11':[0.85,1]}
# Attention il va peut être falloir rajouter un deuxième mois d'avril pour boucler
X_0 = np.ones(2*N,)
begin = data['Day'].min()
end = data['Day'].max()
begin_month = begin.month
begin_year = begin.year
end_month = end.month

threshold = {}
for month in months.keys():
    if datetime.datetime(begin_year,int(month),1) < begin:
        if datetime.datetime(begin_year + 1,int(month),1) < end :
            index_number = []
            current_search_day = datetime.datetime(begin_year +1,int(month),1)
            while len(index_number) == 0 : 
                index_number =data[data['Day'] == current_search_day].index.values.astype(int)
                current_search_day = current_search_day - datetime.timedelta(days = 1)
            threshold[index_number[0]]= months[month]
        else :
            pass
    else:
        if datetime.datetime(begin_year,int(month),1) < end :
            index_number = []
            current_search_day = datetime.datetime(begin_year,int(month),1)
            while len(index_number) == 0 : 
                index_number =data[data['Day'] == current_search_day].index.values.astype(int)
                current_search_day = current_search_day - datetime.timedelta(days = 1)
            threshold[index_number[0]] = months[month]
        else:
            pass

        if datetime.datetime(begin_year+1,int(month)+1,1) < end :
            index_number = []
            current_search_day = datetime.datetime(begin_year+1,int(month)+1,1)
            while len(index_number) == 0 : 
                index_number =data[data['Day'] == current_search_day].index.values.astype(int)
                current_search_day = current_search_day - datetime.timedelta(days = 1)
            threshold[index_number[0]] = months[month]
        else:
            pass
            
data[ 'minV'] = 0
data["maxV"] = 1
Vmax = 100
initial_v = 50

for i in threshold.keys():
    data['minV'].loc[i] = threshold[i][0]
    data['maxV'].loc[i] = threshold[i][1]
data['minV'] = Vmax*data['minV'] 
data['maxV'] = Vmax*data['maxV'] 
plt.plot(data['Day'],data['minV'])
plt.plot(data['Day'],data['maxV'])

Vect_min = np.array(data['minV'])
Vect_max = np.array(data['maxV'])



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

I_diff= np.zeros((N,2*N))

for i in range(0,N):
    I_diff[i,2*i]= -1
    I_diff[i,2*i+1] = +1


triang_sup = np.zeros((N,N))

for i in range(0, N) :
    for j in range(i,N):
        triang_sup[i,j] = 1
initial_vect = initial_v * np.ones(N)
max_vect = Vmax * np.ones(N)
c3 = scipy.optimize.LinearConstraint(triang_sup.transpose()@I_diff,Vect_min- initial_vect ,Vect_max - initial_vect)
#c3 = {'type': 'ineq', 'fun': lambda x : Vmax - volume(-x,initial_v)}

res = minimize(lambda x:profit(x, data['Price']), X_0,method='SLSQP', constraints = [c1,c2,c3],options={'disp': True})
print(-profit(res.x,data['Price']))


volumes=[]
for i in range(0,2*N,2):
    volumes.append(volume(res.x[:i], initial_v))


plt.plot(data['Day'],volumes)

plt.show()
