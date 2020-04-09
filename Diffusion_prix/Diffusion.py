import numpy as np 
import scipy 
import math
import pandas as pd 
import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta, date
import matplotlib.dates as mdates


#Pour juste faire tourner le script sans devoir faire appel à des classes spécifiques
# parser = argparse.ArgumentParser()
# parser.add_argument("nom_fichier", type=str, help='')
# args = parser.parse_args()
# nom = args.nom_fichier


class DiffusionSpot:

    def __init__(self, path1:str, path2:str, skip=[], summer_months=['03', '04', '05', '06', '07', '08'], winter_months=['09', '10', '11', '12', '01', '02'], years=['2019','2020']):
        '''
        Initialise the Diffusion class. The historical data for the creation of the diffusion model
        of spot prices must be in a csv file, placed in the same repository as the script. If not,
        the path to the csv should be specified.
        The model uses different volatility of spot prices, of long-term evolution and different 
        mean-reversion parameters for summer or winter months. The years to specify in the 
        constructor are the ones for which we wish to estimate volatility.
        The skip parameter is just to skip the first few rows of a dataset if they are strings or non 
        integer values.
        Path 1 - spot
        Path 2 - foraward
        '''
        self._dataset = pd.read_csv(path1, header = 0, skiprows = skip)
        if list(self._dataset) != ['Day', 'Price'] and list(self._dataset)!= ['Trading Day', 'Prediction', 'Price']:
            self._dataset.columns = ['Day', 'Prediction', 'Price']  #Set column names
        self.df_forward = pd.read_csv(path2)
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
            somme_diff += abs(price[i] - price[i-1]) #/abs(price[i-1]) if normalized
        return somme_diff/len(price)  #*100 if we want percentage

    def long_volatility(self):
        years = [f'{i}' for i in range(2010, 2021)]
        somme = 0
        df = self.selecting_dataframe(years)
        price = np.array(df['Price'])
        for i in range(1, len(price)):
            somme += abs(price[i] - price[i-1]) #/abs(price[i-1]) if normalized
        return somme/len(price)   #*100 if we want percentage
        
    @property
    def summer_volatility(self):
        self._summer_volatility = self.short_volatility()
        return self._summer_volatility

    @property
    def winter_volatility(self):
        self._winter_volatility = self.short_volatility(False)
        return self._winter_volatility

    def illustrating_mean_reversion(self, summer = True):
        '''
        Function for illustrating before estimating the mean reversion parameter with given historical data.
        Approach supposes the time step is sufficiently small that a naïve description of 
        the U-O process can be taken. We will use a least-squares regression to regress the value
        of the rate of mean-reversion. We plot G_{t+1} - G_{t} = Y against G{t} = X
        '''
        years = self.years
        if summer:
            months = self.summer_months
        else:
            months = self.winter_months
        df = self.selecting_dataframe(years, months)
        price = np.array(df['Price'])
        Y = [price[i] - price[i-1] for i in range(1, len(price))]
        plt.scatter(price[:-1], Y, color = 'r', marker = 'o' )
        plt.legend('G_{t+1} - G_{t} = Y against G{t} = X')
        plt.scatter(price[1:], price[:-1], color = 'b', marker = 'x')
        plt.legend('G_{t+1} = Y against G{t] = X')
        plt.show()

    def mean_reversion(self, summer = True):
        years = self.years
        if summer:
            months = self.summer_months
        else:
            months = self.winter_months
        df = self.selecting_dataframe(years, months)
        price = np.array(df['Price'])
        Y = np.array([price[i] - price[i-1] for i in range(1, len(price))])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(price[:-1], Y )
        return abs(slope)
        

    def fetch_forward(self, start_date):
        '''
        Fetches the right forward price for a given date starting from which we wish to create
        the diffusion spot price model.
        '''
        df = self.df_forward
        upcoming_months = df.loc[df['Trading Day'] == start_date, ['Month+1', 'Month+2', 'Month+3', 'Month+4']]
        return np.array(upcoming_months.values)

    def hurst_exponent(self):
        '''
        The goal of this function is to provide a scalar which allows us to confirm that our
        time series is mean reveting, geometric brownian motion or is trending. This can confirm
        that our mean-reverting pilipovic process is justified and the strength of the rate of
        reversion as well.
        '''
        pass

    def daterange(self, start_date:date, end_date:date):
        '''
        Short function to give a list with incremented dates between a start and end date.
        Start and end dates to given in date format. List of strings will be returned.
        '''
        dates = []
        for n in range(int((end_date - start_date).days)):
            next_date = start_date + timedelta(n)
            dates.append(next_date.strftime('%Y-%m-%d'))
        return dates

    def pilipovic_fixed_forward(self, start_date:str, end_date:str, summer=True):
        '''
        Numerically solves stochastic differential equation of the pilipovic process.
        Here is considered standard brownian motion at each time step. The considered time
        step is a day. The model is run over 4 months so around 122 days (if february is not one of these 
        months). The function will take into account switches between summer and winter.
        '''
        short_vol_sum = self.summer_volatility
        short_vol_win = self.winter_volatility
        mean_reversion_sum = self.mean_reversion()
        mean_reversion_win = self.mean_reversion(False)
        step = 1 #We consider one spot price given out every day. Even week-ends.
        forward_curve = self.fetch_forward(start_date)
        df = self._dataset
        G_0 = np.array((df.loc[df['Day'] == start_date, ['Price']]))[0]
        Spot_curve = [G_0[0]]
        dates = self.daterange(date(int(start_date.split('-')[0]), int(start_date.split('-')[1]), int(start_date.split('-')[2])), date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2])))
        n = len(dates)
        for i in range(1, n):
            if self.string_test(dates[i].split('-')[1], self.summer_months):
                sigma = short_vol_sum
                alpha = mean_reversion_sum
            else:
                sigma = short_vol_win
                alpha = mean_reversion_win
            mean = forward_curve[0, i*4//n]  #should use better date comparison for exact shift in mean value
            G_k = Spot_curve[-1]
            G_k1 = alpha*(mean - G_k) + sigma*np.random.randn() + G_k
            Spot_curve.append(G_k1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(dates, Spot_curve)

    def one_iteration(self, date:str):
        '''
        Input a date, will return a future spot price (one iteration of pilipovic process) based 
        on forward curve and estimated volatilities. Date in %Y-%m-%d.
        '''
        if self.string_test(date.split('-')[1], self.summer_months):
            sigma = self.summer_volatility
            alpha = self.mean_reversion()
        else:
            sigma = self.winter_volatility
            alpha = self.mean_reversion(False)
        forward_curve = self.fetch_forward(date)
        mean = forward_curve[0,0]
        df = self._dataset
        G_0 = np.array((df.loc[df['Day'] == date, ['Price']]))[0]
        return float(alpha*(mean - G_0) + sigma*np.random.randn() + G_0)








    
    
    

    




