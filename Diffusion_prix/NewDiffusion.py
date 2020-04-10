import numpy as np 
import scipy 
import math
import pandas as pd 
import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime
import matplotlib.dates as mdates


class DiffusionSpot:

    def __init__(self, path1:str, path2:str, skip=[], summer_months=[4, 5, 6, 7, 8, 9], winter_months=[10, 11, 12, 1, 2, 3]):
        '''
        Initialise the Diffusion class. The historical data for the creation of the diffusion model
        of spot prices must be in a csv file, placed in the same repository as the script. If not,
        the path to the csv should be specified.
        Summer months in list, same for winter.
        The model uses different volatility of spot prices, of long-term evolution and different 
        mean-reversion parameters for summer or winter months. The years to specify in the 
        constructor are the ones for which we wish to estimate volatility.
        The skip parameter is just to skip the first few rows of a dataset if they are strings or non 
        integer values.
        Path 1 - spot
        Path 2 - forward
        '''
        self._dataset = pd.read_csv(path1, header = 0, skiprows = skip)
        if list(self._dataset) != ['Day', 'Price'] and list(self._dataset) != ['Day', 'Prediction', 'Price']:
            self._dataset.columns = ['Day', 'Prediction', 'Price']  #Set column names
        self._dataset['Day'] = pd.to_datetime(self._dataset['Day'])   #convert all dates to datetime
        self.df_forward = pd.read_csv(path2)
        self.df_forward.rename(columns = {'Trading Day':'Day'}, inplace=True)
        self.df_forward['Day'] = pd.to_datetime(self.df_forward['Day'])
        self.summer_months = summer_months
        self.winter_months = winter_months
        self._summer_volatility = 0
        self._winter_volatility = 0

    def selecting_dataframe(self, start_date:str, end_date:str, spot=True, summer=False, winter=False):
        '''
        Function to select the appropriate dataframe from the original one. Default dataframe 
        from whiwh it extracts is the spot dataframe.
        User inputs a start and end date in between which he wishes to extract data.
        User can also input an optional summer or winter parameter which will extract data solely
        for the summer or winter months between the specified dates.
        Date  %Y-%m-%d format.
        '''
        if spot: 
            df = self._dataset 
        else: 
            df = self.df_forward
        start_date, end_date = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')
        df = df.loc[(df['Day'] >= start_date) & (df['Day']<= end_date)]
        if summer:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.summer_months)]
        if winter:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.winter_months)]
        df.dropna(inplace = True) #In case there are missing values
        return df

    def short_volatility(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        This function returns the volatilty of a certain range of spot prices over the given dataframe.
        Optional parameter allow the user to select only summer or winter months for the volatility.
        The volatility is calculated taking the standard deviation from the series of log of the spot prices.
        '''
        df = self.selecting_dataframe(start_date, end_date, True, summer = summer, winter = winter)
        price = np.array(df['Price'])
        somme_diff = 0
        if len(price)!=0:
            for i in range(1, len(price)):
                somme_diff += abs(price[i] - price[i-1]) #/abs(price[i-1]) if normalized
            return somme_diff/len(price)  #*100 if we want percentage
        else:
            return 0

    def illustrating_mean_reversion(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        Function for illustrating before estimating the mean reversion parameter with given historical data.
        Approach supposes the time step is sufficiently small that a naïve description of 
        the U-O process can be taken. We will use a least-squares regression to regress the value
        of the rate of mean-reversion. We plot G_{t+1} - G_{t} = Y against G{t} = X
        '''
        df = self.selecting_dataframe(start_date, end_date, True, summer = summer, winter = winter)
        price = np.array(df['Price'])
        Y = [price[i] - price[i-1] for i in range(1, len(price))]
        plt.scatter(price[:-1], Y, color = 'r', marker = 'o' )
        plt.legend('G_{t+1} - G_{t} = Y against G{t} = X')
        plt.scatter(price[1:], price[:-1], color = 'b', marker = 'x')
        plt.legend('G_{t+1} = Y against G{t] = X')
        plt.show()

    def mean_reversion(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        Calculate the mean reversion parameter in the pilipovic process on a given dataframe.
        '''
        df = self.selecting_dataframe(start_date, end_date, True, summer = summer, winter = winter)
        price = np.array(df['Price'])
        if len(price)!=0:
            Y = np.array([price[i] - price[i-1] for i in range(1, len(price))])
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(price[:-1], Y )
            return abs(slope)
        else:
            return 0
    
    def fetch_forward(self, start_date):
        '''
        Fetches the right forward price for a given date starting from which we wish to create
        the diffusion spot price model.
        '''
        df = self.df_forward
        upcoming_months = df.loc[df['Day'] == datetime.strptime(start_date, '%Y-%m-%d'), ['Month+1', 'Month+2', 'Month+3', 'Month+4']]
        return np.array(upcoming_months.values)

    def daterange(self, start_date:str, end_date:str):
        '''
        Short function to give a list with incremented dates between a start and end date.
        Start and end dates to given in date format. List of strings will be returned.
        '''
        dates = []
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        for n in range(int((end_date - start_date).days)):
            next_date = start_date + timedelta(n)
            dates.append(next_date)
        return dates

    def pilipovic_fixed_forward(self, start_date:str, end_date:str, end_date_sim:str):
        '''
        Numerically solves stochastic differential equation of the pilipovic process.
        Here is considered standard brownian motion at each time step. The considered time
        step is a day. The model is run between end_date and end_date_sim. 
        Numerical parameters are estimated between start_date and end_date which is historical data.
        The function takes into account switches between summer and winter in considered time period.
        '''
        df = self.selecting_dataframe(start_date, end_date)
        short_vol_sum = self.short_volatility(start_date, end_date, summer=True)
        short_vol_win = self.short_volatility(start_date, end_date, winter=True)
        mean_reversion_sum = self.mean_reversion(start_date, end_date, summer=True)
        mean_reversion_win = self.mean_reversion(start_date, end_date, summer=False)
        forward_curve = self.fetch_forward(end_date)
        G_0 = df['Price'].to_list()[-1]
        Spot_curve = [G_0]
        dates = self.daterange(end_date, end_date_sim)
        n = len(dates)
        for i in range(1, n):
            if dates[i].month in self.summer_months:
                if short_vol_sum == 0 or mean_reversion_sum == 0:
                    alpha = mean_reversion_win
                    sigma = short_vol_win
                else:
                    sigma = short_vol_sum
                    alpha = mean_reversion_sum
            else:
                if short_vol_win == 0 or mean_reversion_sum == 0:
                    alpha = mean_reversion_sum
                    sigma = short_vol_sum
                else:
                    sigma = short_vol_win
                    alpha = mean_reversion_win
            mean = forward_curve[0, i*4//n]  #should use better date comparison for exact shift in mean value
            G_k = Spot_curve[-1]
            G_k1 = alpha*(mean - G_k) + sigma*np.random.randn() + G_k
            print(G_k1, alpha, mean, sigma, G_k)
            Spot_curve.append(G_k1)
        plt.plot(dates, Spot_curve)
