import numpy as np
class Matrices :
    def __init__(self,N):
        self.N = N
    
    def I_sell_constructor(self):
        I_sell = np.zeros((self.N,2*self.N))
        for i in range(0,self.N):
            I_sell[i,2*i]=1
        return I_sell
    
    I_sell = property(I_sell_constructor)

    def I_buy_constructor(self):
        I_buy = np.zeros((self.N,2*self.N))
        for i in range(self.N):
            I_buy[i,2*i+1]=1
        return I_buy
    
    I_buy = property(I_buy_constructor)

    def I_diff_constructor(self):
        I_diff= np.zeros((self.N,2*self.N))
        for i in range(0,self.N):
            I_diff[i,2*i]= -1
            I_diff[i,2*i+1] = +1
        return I_diff

    I_diff = property(I_diff_constructor)
   
    def triang_inf_constructor(self):
        triang_sup = np.zeros((self.N,self.N))
        for i in range(0, self.N) :
            for j in range(i,self.N):
                triang_sup[i,j] = 1
        return triang_sup.transpose()
    
    triang_inf = property(triang_inf_constructor)

m = Matrices(4)

        
