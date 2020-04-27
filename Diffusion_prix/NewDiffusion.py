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

    def __init__(self, path1:str, path2:str, skip=[], summer_months=[4, 5, 6, 7, 8, 9], winter_months=[10, 11, 12, 1, 2, 3], forward_diffusion=True):
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
        Weekends - included or not in the dataframe. 
        Forward diffusion - if the diffusion model is around forward pricing (True) or historical mean (False).
        '''
        self._dataset = pd.read_csv(path1, header = 0, skiprows = skip)
        if list(self._dataset) != ['Day', 'Price'] and list(self._dataset) != ['Day', 'Prediction', 'Price']:
            self._dataset.columns = ['Day', 'Prediction', 'Price']  #Set column names
        self._dataset['Day'] = pd.to_datetime(self._dataset['Day'])   #convert all dates to datetime
        if forward_diffusion:
            self.df_forward = pd.read_csv(path2)
            self.df_forward.rename(columns = {'Trading Day':'Day'}, inplace=True)
            self.df_forward['Day'] = pd.to_datetime(self.df_forward['Day']) #convert dates to datetime
        self.summer_months = summer_months
        self.winter_months = winter_months
        self._weekends = True   #property will check automatically if week-ends are included or not in the initial dataframe
        self.forward_diffusion = forward_diffusion

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
        if not self.weekends:                    #check if user input date is not a weekend if weekends are not included in df
            if start_date.weekday() > 4 or end_date.weekday() > 4:
                raise ValueError("The dataframe does not have WE, do not input start or end date as WE.")
        df = df.loc[(df['Day'] >= start_date) & (df['Day']<= end_date)]
        if summer:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.summer_months)]
        if winter:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.winter_months)]
        df.dropna(inplace = True) #In case there are missing values
        return df

    def volatility(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        This function returns the volatilty of a certain range of spot prices over the given dataframe.
        Optional parameter allow the user to select only summer or winter months for the volatility.
        The volatility is calculated taking the standard deviation from the series of log of the spot prices.
        Divide by len(m)-1 to produce an unbiased estimator.
        '''
        df = self.selecting_dataframe(start_date, end_date, True, summer = summer, winter = winter)
        price = np.array(df['Price'])
        if len(price)!=0:
            series = np.array([np.log(price[i]/price[i-1]) for i in range(1, len(price))])
            mean = series.mean()
            series = (series - mean)**2
            variance = (1/(len(price)-1))*sum(series)  #unbiased estimator
            return np.sqrt(variance)
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
            return 0  #if df to estimate mean_reversion parameter does not have data 
    
    def fetch_forward(self, start_date):
        '''
        Fetches the right forward price for a given date starting from which we wish to create
        the diffusion spot price model.
        Be careful, unlike spot prices in certain datasets, there are no forward prices issued on week ends.
        If given start_date is a week-end, error is raised.
        '''
        df = self.df_forward
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if start_date.weekday() <= 4:
            upcoming_months = df.loc[df['Day'] == start_date, ['Month+1', 'Month+2', 'Month+3', 'Month+4']]
            return np.array(upcoming_months.values)[0]
        else:
            raise ValueError("The given start_date is a week-end. Please input a week day for the simulation!")

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
            if self.weekends:
                 dates.append(next_date)
            else: 
                if next_date.weekday() <= 4:
                    dates.append(next_date)  
        return dates


    def pilipovic_fixed_parameters(self, start_date_long:str, end_date_long:str, start_date:str, end_date:str):
        '''
        Calculating both short term (summer and winter) and long term volatility can be computationally intensive.
        This function calculates the parameters which will be used in the fixed forward or mean-reverting pilipovic process.
        '%Y-%m-%d' format for dates

        '''
        vol_sum = self.volatility(start_date, end_date, summer=True)
        vol_win = self.volatility(start_date, end_date, winter=True)
        mean = 0
        long_term_vol = 0
        if vol_sum == 0:  #if we don't have enough data we might encounter a problem with volatility estimation
            vol_sum = vol_win
        elif vol_win == 0:
            vol_win = vol_sum
        mean_reversion_sum = self.mean_reversion(start_date, end_date, summer=True)
        mean_reversion_win = self.mean_reversion(start_date, end_date, winter=True)
        if mean_reversion_sum == 0:
            mean_reversion_sum = mean_reversion_win
        elif mean_reversion_win == 0:
            mean_reversion_sum = mean_reversion_sum
        if not self.forward_diffusion: #to avoid expensive computation if not necessary
            mean = np.mean(np.array(self.selecting_dataframe(start_date_long, end_date_long)['Price']))
            long_term_vol = self.volatility(start_date_long, end_date_long)
        return vol_sum, vol_win, mean_reversion_sum, mean_reversion_win, mean, long_term_vol

    def pilipovic_fixed_forward(self, end_date, end_date_sim, vol_sum, vol_win, mean_reversion_sum, mean_reversion_win, mean, long_term_vol, return_forward=False):
        '''
        Numerically solves stochastic differential equation of the pilipovic process.
        Here is considered standard brownian motion at each time step. The considered time
        step is a day. The model is run between end_date and end_date_sim. 
        The function takes into account switches between summer and winter in considered time period.
        as the "mean" or the parameter which we revert back to.
        Return forward - Boolean to indicate if return of forward curve is needed. 
        If construction parameter return forward_diffusion is False, then mean is returned.
        '''
        df = self.selecting_dataframe(end_date, end_date)
        G_0 = df['Price'].to_list()[-1]  #simulation starts from the last day used for the estimation of short parameters
        dates = self.daterange(end_date, end_date_sim)
        Spot_curve = [G_0]
        n = len(dates)
        volatilities = [vol_sum if (dates[i].month in self.summer_months) else vol_win for i in range(n)]
        mean_reversions = [mean_reversion_sum if (dates[i].month in self.summer_months) else mean_reversion_win for i in range(n)]
        if self.forward_diffusion: #diffusion autour de forward fixe
            forward_curve = self.fetch_forward(end_date)
            for i in range(1, n):
                forward = forward_curve[i*4//n]  #will be more complicated if forward data is in a different format than webscraped one
                G_k = Spot_curve[-1]
                G_k1 = mean_reversions[i]*(forward- G_k) + volatilities[i]*np.random.randn() + G_k  # N(0,1)
                Spot_curve.append(G_k1)
        else:                         #diffusion autour d'une moyenne accompagnée d'un mouvement determiné par la volatilité à long terme
            means = [mean]
            for i in range(1, n):
                 G_k = Spot_curve[-1]
                 mean_k = means[-1]
                 G_k1 = mean_reversions[i]*(mean_k - G_k) + volatilities[i]*np.random.randn() + G_k 
                 mean_k1 = mean_k + long_term_vol*np.random.randn()
                 Spot_curve.append(G_k1)
                 means.append(mean_k1)
        if return_forward:
            return Spot_curve, forward_curve
        else:
            return Spot_curve

    def multiple_price_scenarios(self, start_date_long:str, end_date_long:str, start_date:str, end_date:str, end_date_sim:str, n:int):
        '''
        Generates n number of spot price scenarios using a pilipovic process for the evolution 
        dynamics of the spot price. Forward price is based on future curve at start_date.
        Date format is %Y-%m-%d. Gives table with all spot curves and a curve of mean calculated 
        over all spot curves.
        '''
        moyenne = []
        vol_sum, vol_win, mean_reversion_sum, mean_reversion_win, mean, long_term_vol = self.pilipovic_fixed_parameters(start_date_long, end_date_long, start_date, end_date)
        first_simul, forward_curve = self.pilipovic_fixed_forward(end_date, end_date_sim, vol_sum, vol_win, mean_reversion_sum, mean_reversion_win, mean, long_term_vol, return_forward=True)
        tab = [first_simul]
        for k in range(1, n):
            tab.append(self.pilipovic_fixed_forward(end_date, end_date_sim, vol_sum, vol_win, mean_reversion_sum, mean_reversion_win, mean, long_term_vol))
        for i in range(len(tab[0])):
            moyenne.append(sum(tab[k][i] for k in range(len(tab)))/len(tab))
        return np.array(tab), np.array(moyenne), forward_curve

    def show_multiple(self, start_date_long:str, end_date_long:str, start_date:str, end_date:str, end_date_sim:str, n:int):
        '''
        Function to display the multiple price scenarios created as well as the mean curve
        and forward prices associated.
        '''
        tab, moyenne, forward_curve = self.multiple_price_scenarios(start_date_long, end_date_long, start_date, end_date, end_date_sim, n)
        dates = self.daterange(end_date, end_date_sim)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        for i in range(len(tab)):
            ax1.plot(dates, tab[i], lw=1)
        ax1.xaxis_date()
        fig.autofmt_xdate()
        ax1.set_title(f'{n} Spot price scenarios')
        ax1.set_ylabel("Spot Price (€/MWh)")
        if self.forward_diffusion:
            curve = [forward_curve[i*4//len(tab[0])] for i in range(len(tab[0]))]
            ax2.plot(dates, curve, label='Prix forward', lw=2)
        else:
            curve = [forward_curve for i in range(len(tab[0]))]
            ax2.plot(dates, curve, label='Mean', lw=2)
        ax2.plot(dates, moyenne, lw=2, label='Moyenne prix spot')
        ax2.set_ylabel('€/MWh')
        ax2.set_title('Moyenne des scénarios spot en fonction du temps')
        plt.show()

    @property
    def weekends(self):
        '''
        Function to detect if w-e are included or not in the dataframe
        '''
        df = self._dataset['Day'][:10]
        self._weekends = False
        for element in df:
            if element.weekday() > 4:
                self._weekends = True
        return self._weekends

    # THIS FUNCTION NEEDS TO BE CHANGED
    # def one_iteration(self, start_date:str, end_date:str, simul_date:str, spot_price = None):
    #     '''
    #     Input dates for estimation of parameters and date of simulation, will return a future spot price (one iteration of pilipovic process) based 
    #     on forward curve and estimated volatilities. Date in %Y-%m-%d. 
    #     If spot price not provided, will just take the spot_price at end day as the current spot price based upon which
    #     next step is calculated.
    #     '''
    #     df = self.selecting_dataframe(start_date, end_date)
    #     vol_sum = self.volatility(start_date, end_date, summer=True)
    #     vol_win = self.volatility(start_date, end_date, winter=True)
    #     if vol_sum == 0:  #if we don't have enough data we might encounter a problem with volatility estimation
    #         vol_sum = vol_win
    #     elif vol_win == 0:
    #         vol_win = vol_sum
    #     mean_reversion_sum = self.mean_reversion(start_date, end_date, summer=True)
    #     mean_reversion_win = self.mean_reversion(start_date, end_date, summer=False)
    #     if mean_reversion_sum == 0:
    #         mean_reversion_sum = mean_reversion_win
    #     elif mean_reversion_win == 0:
    #         mean_reversion_sum = mean_reversion_sum
    #     forward_curve = self.fetch_forward(end_date)
    #     mean = forward_curve[int(simul_date.split('-')[1]) - int(end_date.split('-')[1])]
    #     if simul_date.split('-')[1] in self.summer_months:
    #         sigma = vol_sum
    #         alpha = mean_reversion_sum
    #     else:
    #         sigma = vol_win
    #         alpha = mean_reversion_win
    #     if not spot_price:
    #         G_0 = df['Price'].to_list()[-1]
    #     else:
    #         G_0 = spot_price
    #     return float(alpha*(mean - G_0) + sigma*np.random.randn() + G_0)