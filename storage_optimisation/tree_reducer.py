import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# tree = pd.read_csv('test.csv', delimiter = ';', decimal = ',', header = 0)
# tree = np.array(tree)

# plt.plot(tree)
# n_scen = len(tree[0])
# print(n_scen)

# print('tree_prob', tree_prob)
#print(tree)
class Scenario_builder():
    def __init__(self):
        pass
    
    def data_loader(self, file):
        tree = pd.read_csv(file, delimiter = ';', decimal = ',', header = 0)
        self.tree = np.array(tree)
        self.tree_prob  = np.ones(self.n_scen)/self.n_scen
        plt.plot(self.tree, color = 'b')
        plt.show()

    def distance_scen_pair(self, i, j,t_max):
        return sum(abs(self.tree.transpose()[i][:t_max]-self.tree.transpose()[j][:t_max]))
    
    n_scen = property(lambda self : len(self.tree[0]))
    T = property(lambda self: len(self.tree))
    def mat_distance_scen(self, T_max):
        index = {i for i in range(self.n_scen)}
        C_K_J = []
        for k in range(self.n_scen):
            C_J = []
            for j in range(self.n_scen):
                C_J.append(self.distance_scen_pair(k,j,T_max))
            C_K_J.append(C_J)
        C_K_J = np.array(C_K_J)
        self.C_K_J = C_K_J

    def backward_reduction(self, n_deletion, T_max = False):
        if T_max == False:
            T_max = self.T
        else:
            pass
        index = {i for i in range(self.n_scen)}
        J = set()
        C_LL = np.zeros(self.n_scen)
        self.mat_distance_scen(T_max)
        for l in range(self.n_scen):
            C_LL[l] = min(self.C_K_J[l][list(index.difference(J|{l}))])
        z_L = self.tree_prob*C_LL
        l = np.argmin(z_L)
        J = J|{l}
        #Redondance avec l'initialisation à voir 
        while len(J)< n_deletion:
            Z_L = []
            for  l in index.difference(J):
                C_KL = []
                for k in (J|{l}):
                    c_kl = min(self.C_K_J[k][list(index.difference(J|{l}))])
                    C_KL.append(c_kl)
                C_KL = np.array(C_KL)
                z_l = sum(self.tree_prob[np.array(list(J|{l}))]*C_KL)
                Z_L.append(z_l) # FAire des test sur la taille des listes qui sont crées
            l = list(index.difference(J))[np.argmin(Z_L)]
            J = J|{l}
        return J 
    

    def backward_reduction_iter(self, index, T_max = False):

        if T_max == False:
            T_max = self.T
        else:
            pass
        J = set()
        C_LL = np.zeros(self.n_scen)
        self.mat_distance_scen(T_max)
        while len(J)< self.n_scen:
            Z_L = []
            for  l in index.difference(J):
                C_KL = []
                for k in (J|{l}):
                    c_kl = min(self.C_K_J[k][list(index.difference(J|{l}))])
                    C_KL.append(c_kl)
                C_KL = np.array(C_KL)
                z_l = sum(self.tree_prob[np.array(list(J|{l}))]*C_KL)
                Z_L.append(z_l) # FAire des test sur la taille des listes qui sont crées
            l = list(index.difference(J))[np.argmin(Z_L)]
            J = J|{l}
            yield J, min(Z_L)

    def scenario_deletion(self, n_deletion ):
        index = {i for i in range(self.n_scen)}
        tree_prob = self.tree_prob
        b = sum(self.tree_prob)
        J = self.backward_reduction(n_deletion)
        for i in J :
            j = np.argmin(self.C_K_J[i][list(index.difference(J))])
            j = list(index.difference(J))[j]
            # assert not j in J : A passer en test 
            tree_prob[j] += tree_prob[i]

        tree_t = self.tree.transpose()[list(index.difference(J))]
        self.tree = tree_t.transpose()
        self.tree_prob = tree_prob[list(index.difference(J))]
        a = sum(self.tree_prob)
        print(a-b)

    def plot_tree(self):
        plt.plot(self.tree, color = 'r')
        plt.show()

    def scenario_tree_construction(self,epsilon):
        index = {i for i in range(self.n_scen)}
        boundings = {}
        for pas in range(1,self.T):
            ## Reduction 
            
            iter_J = self.backward_reduction_iter(index,self.T-pas+1)
            J,Z = next(iter_J)
            while Z < epsilon:
                J,Z = next(iter_J)
            # print('k',pas)
            # print('I',index)
            # print('J',J)
            
            # scenario bundling
            for j in J :
                i = np.argmin(self.C_K_J[j][list(index.difference(J))])
                i= list(index.difference(J))[i]
                self.tree_prob[i] += self.tree_prob[j]
                self.tree[:self.T-pas+1,j] = self.tree[:self.T-pas+1,i]
                transfer_bound = [j]
                if j in boundings.keys():
                    bounds = boundings[j]
                    for q in bounds:
                        self.tree[:self.T-pas+1,q] = self.tree[:self.T-pas+1,i]
                    transfer_bound.append(j)
                if i in boundings.keys():
                    boundings[i] = boundings[i]+transfer_bound
                else : 
                    boundings[i] = transfer_bound
            plt.plot(self.tree)
            plt.show()
            for i in range(2,self.n_scen):
                pass
            index = index.difference(J)
            if len(index) <= 1 :
                break





builder = Scenario_builder()
plt.show()
builder.data_loader('test.csv')
builder.scenario_tree_construction(7)


builder.plot_tree()
