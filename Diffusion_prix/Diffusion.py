import numpy as np 
import scipy 
import math
import pandas as pd 
import argparse


#Pour juste faire tourner le script sans devoir faire appel à des classes spécifiques
# parser = argparse.ArgumentParser()
# parser.add_argument("nom_fichier", type=str, help='')
# args = parser.parse_args()
# nom = args.nom_fichier


class DiffusionSpot:

    def __init__(self, path:str, summer_months=['03', '04', '05', '06', '07', '08'], winter_months=['09', '10', '11', '12', '01', '02'], years=['2019','2020']):
        '''
        Initialise the Diffusion class. The historical data for the creation of the diffusion model
        of spot prices must be in a csv file, placed in the same repository as the script. If not,
        the path to the csv should be specified.
        The model uses different volatility of spot prices, of long-term evolution and different 
        mean-reversion parameters for summer or winter months. The years to specify in the 
        constructor are the ones for which we wish to estimate volatility
        '''
        self._dataset = pd.read_csv(path, header = 0, skiprows = [1,2,3,4])
        if list(self._dataset) != ['Day', 'Price']:
            self._dataset.columns = ['Day', 'Price']  #Set column names
        self.summer_months = summer_months
        self.winter_months = winter_months
        self.years = years
        self._summer_volatility = 0
        self._winter_volatility = 0

    def string_test(self, string, list_str):
        '''
        Short test needed in volatility method to make code clearer.
        Tests if string is in list of string (months or years).
        '''
        return any([string == list_str[i] for i in range(len(list_str))])

    def selecting_dataframe(self, years, months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
        '''
        Function to select the appropriate dataframe from the original one. User can choose months
        and years he/she wishes to extract from the original dataframe when using the function. It 
        will be used to extract the summer and winter parameters to create our diffusion model.
        '''
        df = self._dataset
        rows = []
        for i in range(len(df)):
            if self.string_test(np.array(df['Day'])[i].split('-')[1], months)  and self.string_test(np.array(df['Day'])[i].split('-')[0], years):
                rows.append(i)
        df1 = df[df.index.isin(rows)]
        df1.dropna(inplace = True) #In case there are missing values
        return df1 

    def short_volatility(self, summer = True):
        '''
        This function returns the volatilty of a certain range of spot prices. This is calculated
        over specified months, summer or winter and for t. 
        '''
        if summer:
            months = self.summer_months
        else:
            months = self.winter_months
        years = self.years
        df = self.selecting_dataframe(years, months)
        price = np.array(df['Price'])
        somme_diff = 0
        for i in range(1, len(price)):
            somme_diff += abs(price[i] - price[i-1])/abs(price[i-1])
        return somme_diff/len(price)*100

    def long_volatility(self):
        years = [f'{i}' for i in range(2010, 2021)]
        somme = 0
        df = self.selecting_dataframe(years)
        price = np.array(df['Price'])
        for i in range(1, len(price)):
            somme += abs(price[i] - price[i-1])/abs(price[i-1])
        return somme/len(price)*100
        
    @property
    def summer_volatility(self):
        self._summer_volatility = self.short_volatility()
        return self._summer_volatility

    @property
    def winter_volatility(self):
        self._winter_volatility = self.short_volatility(False)
        return self._winter_volatility

    def pilipovic(self, n, summer = True, t_fin:int, t_ini = 0):
        '''
        Numerically solves stochastic differential equation of the pilipovic process.
        Here is considered standard , uncorellated brownian motion
        '''
        if summer:
            short_vol = self.summer_volatility
        else:
            short_vol = self.winter_volatility
        mean_reversion = 
        step = (t_fin - t_ini)/n




    
    
    

    




