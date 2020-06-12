import numpy as np 
import scipy 
import math
import pandas as pd 
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta, MO


class DiffusionSpot:

    def __init__(self, path1:str, path2:str, skip=[], summer_months=[4, 5, 6, 7, 8, 9], winter_months=[10, 11, 12, 1, 2, 3], forward_diffusion=True):
        '''
        Initialise the Diffusion class.
        The model uses different volatility of spot prices, of long-term evolution and different 
        mean-reversion parameters for summer or winter months. 
        Inputs:
        Path 1 - spot
        Path 2 - forward
        The skip parameter is just to skip the first few rows of a dataset if they are strings or non 
        integer values. 
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
            self.df_forward.drop(columns = ['Month+5', 'Calendar+3'], inplace = True)
        self.summer_months = summer_months
        self.winter_months = winter_months
        self._weekends = True   #property will check automatically if week-ends are included or not in the initial dataframe
        self.forward_diffusion = forward_diffusion

    def selecting_dataframe(self, start_date:str, end_date:str, spot=True, summer=False, winter=False, remove_weekends = False):
        '''
        Function to select the appropriate dataframe from the original one. 
        Inputs:
        Start and end date in between which he wishes to extract data. Date  %Y-%m-%d format.
        User can also input an optional summer or winter parameter which will extract data solely
        for the summer or winter months between the specified dates.
        Remove_weekends - user may want to remove week end values
        Output:
        Desired dataframe
        '''
        if spot: 
            df = self._dataset 
        else: 
            df = self.df_forward
        start_date, end_date = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')
        if remove_weekends:
            df = df.loc[df['Day'].apply(lambda x:x.weekday() <= 4)]
        if not self.weekends or remove_weekends:                    #check if user input date is not a weekend if weekends are not included in df
            if start_date.weekday() > 4:
                raise ValueError(f"The dataframe does not have WE, do not input {start_date}.")
            if end_date.weekday() > 4:
                raise ValueError(f"The dataframe does not have WE, do not input {end_date}.")
        df = df.loc[(df['Day'] >= start_date) & (df['Day']<= end_date)]
        if summer:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.summer_months)]
        if winter:
            df = df.loc[df['Day'].apply(lambda x:x.month in self.winter_months)]
        df = df.dropna() #In case there are missing values
        return df

    def volatility(self, start_date:str, end_date:str, summer=False, winter=False, annualized=False, remove_weekends=False, mean_include=False):
        '''
        This function returns the volatilty of a certain range of spot prices.
        Inputs:
        Dates - %Y-%m-%d 
        See selecting_dataframe for summer, winter and remove_weekends explanation.
        Annualized = True is for calculating long_term volatility of historical mean
        mean_include = boolean, True if Output should also have mean of spot time series
        Output:
        Volatility as taken to be the standard deviation of the series of daily log changes of spot or forward price.
        '''
        df = self.selecting_dataframe(start_date, end_date, spot = True, summer = summer, winter = winter, remove_weekends = remove_weekends)
        price = np.array(df['Price'])
        spot_mean = price.mean()
        if len(price)!=0:
            series = np.array([np.log(price[i]/price[i-1]) for i in range(1, len(price))])
            mean = series.mean()
            series = (series - mean)**2
            variance = (1/(len(price)-1))*sum(series)  #unbiased estimator
            if annualized:
                if mean_include:
                    return np.sqrt(variance), spot_mean
                else:
                    return np.sqrt(variance) 
            else:
                if mean_include:
                    return np.sqrt(variance)*np.sqrt(len(price)), spot_mean
                else:
                    return np.sqrt(variance)*np.sqrt(len(price))
        else:
            return 0

    def forward_volatility(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        Inputs - dates %Y-%m-%d and summer, winter
        Outputs - forward volatilities for each given time to maturity in the forward dataframe.
        In our case we are working with webscraped format  i.e 4 times to maturity.
        For other format, pre formatting will be necessary.
        '''
        df = self.selecting_dataframe(start_date, end_date, summer = summer, winter = winter, spot = False)
        price = np.array(df[['Month+1', 'Month+2', 'Month+3', 'Month+4']])
        series = np.array([np.array([np.log(price[i, j]/price[i-1, j]) for i in range(1, len(price))]) for j in range(4)])
        means = np.array([[series[i].mean() for k in range(len(series[0]))] for i in range(len(series))])
        series = (series - means)**2
        variances = np.array([(1/(len(price)-1))*sum(series[i]) for i in range(len(series))])
        return np.array([np.sqrt(variances[i])*np.sqrt(len(price)) for i in range(len(variances))])
        

    def illustrating_mean_reversion(self, start_date:str, end_date:str, summer=False, winter=False):
        '''
        Function for illustrating before estimating the mean reversion parameter with given historical data.
        Approach supposes the time step is sufficiently small that a naïve description of 
        the U-O process can be taken. We will use a least-squares regression to regress the value
        of the rate of mean-reversion. We plot G_{t+1} - G_{t} = Y against G{t} = X
        '''
        df = self.selecting_dataframe(start_date, end_date, summer = summer, winter = winter)
        price = np.array(df['Price'])
        Y = [price[i] - price[i-1] for i in range(1, len(price))]
        plt.scatter(price[:-1], Y, color = 'r', marker = 'o' )
        plt.legend('G_{t+1} - G_{t} = Y against G{t} = X')
        plt.scatter(price[1:], price[:-1], color = 'b', marker = 'x')
        plt.legend('G_{t+1} = Y against G{t] = X')
        plt.show()
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(price[:-1], Y)
        return slope, intercept, r_value, p_value, std_err 

    def mean_reversion(self, start_date:str, end_date:str, summer=False, winter=False, allparams=False, spot=True, remove_weekends=False):
        '''
        Inputs - dates and summer, winter as selecting_dataframe
        allparams is for checking p_value and checking the regression is reliable
        Outputs - mean reversion parameter in the pilipovic process of selected timeseries
        '''
        df = self.selecting_dataframe(start_date, end_date, True, summer = summer, winter = winter, remove_weekends = remove_weekends)
        price = np.array(df['Price'])
        if len(price)!=0:
            Y = np.array([price[i] - price[i-1] for i in range(1, len(price))])
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(price[:-1], Y )
            if allparams:
                return slope, intercept, r_value, p_value, std_err
            else:
                return abs(slope)
        else:
            return 0  #if df to estimate mean_reversion parameter does not have data 
    

    def forward_mean_reversion(self, start_date:str,end_date:str, summer=False, winter=False):
        '''
        Same principle as for forward volatilities.
        Outputs 4 mean_reversion parameters and 4 historical means
        '''
        df = self.selecting_dataframe(start_date, end_date, summer = summer, winter = winter, spot = False)
        price = np.array([np.array(df['Month+1']), np.array(df['Month+2']), np.array(df['Month+3']), np.array(df['Month+4'])])
        Y = [np.array([element[i] - element[i-1] for i in range(1, len(price[0]))]) for element in price]
        regress = [abs(scipy.stats.linregress(price[i][:-1], Y[i]).slope) for i in range(len(Y))]
        means = np.array([price[i].mean() for i in range(len(price))])
        return regress, means

    def fetch_forward(self, start_date):
        '''
        Inputs - start_date of fixed_forward diffusion model
        Outputs - forward prices for the next 4 months
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

    def daterange(self, start_date:str, end_date:str, remove_weekends=False):
        '''
        Short function to give a list with incremented dates between a start and end date.
        Start and end dates to given in date format. List of strings will be returned.
        pd.date_range is a good alternative
        '''
        dates = []
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        for n in range(int((end_date - start_date).days)):
            next_date = start_date + timedelta(n)
            if remove_weekends or not self.weekends:
                if next_date.weekday() <= 4:
                    dates.append(next_date)  
            else: 
                dates.append(next_date)  
        return dates

    def date_without_weekends(self, start_date, n):
        '''
        Inputs - start_date in datetime format. Number of days to add without counting weekends
        Outputs - end_date
        This will be useful for dynamic_forward diffusion model used on dataframes without weekends
        '''
        current_date = start_date
        days_to_add = n
        while days_to_add > 0:
            current_date += timedelta(days=1)
            if current_date.weekday() <= 4:
                days_to_add -= 1
            else:
                continue
        return current_date

    def previous_date(self, date, n:int):
        '''
        Input - date datetime format, integer for the number months. Will deal with str
        Output - the date corresponding to date - n months in str format.
        Will always output a day of the week. If week-end will give the following monday
        Just for cleaner random_forest_dataset_spot code.
        '''
        if type(date) is str:
            date = datetime.strptime(date, '%Y-%m-%d')
        new_date = date - relativedelta(months = n)
        if new_date.weekday() > 4:
            new_date = new_date + relativedelta(weekday = MO(+1)) #Next monday
        new_date = new_date.strftime('%Y-%m-%d')
        return new_date

    def volatilities_range(self, start_date, end_date, start_date_sim, end_date_sim, remove_weekends=False):
        '''
        Inputs - start and end dates for simulation. Start and end dates for estimating volatility
        Outputs - array with summer and winter volatilities or each day of diffusion model
        '''
        dates = self.daterange(start_date_sim, end_date_sim, remove_weekends = remove_weekends)
        self.vol_sum = self.volatility(start_date, end_date, summer=True, remove_weekends = remove_weekends, mean_include=False)
        self.vol_win = self.volatility(start_date, end_date, winter=True, remove_weekends = remove_weekends, mean_include=False)
        if self.vol_sum == 0:  #if we don't have enough data we might encounter a problem with volatility estimation
            self.vol_sum = self.vol_win  #Passed as attributes to print them in our later scripts
        elif self.vol_win == 0:
            self.vol_win = self.vol_sum
        return np.array([self.vol_sum if (dates[i].month in self.summer_months) else self.vol_win for i in range(len(dates))])
    
    
    
    def mean_reversion_range(self, start_date, end_date, start_date_sim, end_date_sim, remove_weekends=False):
        '''
        Inputs - start and end dates for simulation. . Start and end dates for estimating mean reversion
        Outputs - array with summer and winter mean reverision parameters for each day of diffusion model
        '''
        dates = self.daterange(start_date_sim, end_date_sim, remove_weekends = remove_weekends)
        self.mean_reversion_sum = self.mean_reversion(start_date, end_date, summer=True, remove_weekends = remove_weekends)
        self.mean_reversion_win = self.mean_reversion(start_date, end_date, winter=True, remove_weekends = remove_weekends)
        if self.mean_reversion_sum == 0:
            self.mean_reversion_sum = self.mean_reversion_win
        elif self.mean_reversion_win == 0:
            self.mean_reversion_win = self.mean_reversion_sum   
        return  np.array([self.mean_reversion_sum if (dates[i].month in self.summer_months) else self.mean_reversion_win for i in range(len(dates))])


    def pilipovic_fixed_parameters(self, start_date_long:str, end_date_long:str, start_date:str, end_date:str, end_date_sim:str, remove_weekends = False):
        '''
        Calculating both short term (summer and winter) and long term volatility can be computationally intensive.
        This function calculates the parameters which will be used in every simulation of fixed forward or mean-reverting pilipovic process.
        '%Y-%m-%d' format for dates. 
        Outputs:
        arrays of volatilities and mean reversion parameters to be used for each time step of the simulation.
        an array with a scenario of mean diffusion, that is a deviation of long term mean based on long term volatility of the market
        length of simulation to avoid computing the dates every time pilipovic_fixed_forward is called.
        '''
        dates = self.daterange(end_date, end_date_sim, remove_weekends=remove_weekends)
        n = len(dates)
        means = [] 
        volatilities = self.volatilities_range(start_date, end_date, end_date, end_date_sim, remove_weekends=remove_weekends)
        mean_reversions = self.mean_reversion_range(start_date, end_date, end_date, end_date_sim, remove_weekends=remove_weekends)
        if self.forward_diffusion:
            self.forward_curve = self.fetch_forward(end_date)  #needs to be accessed by multiple functions
        else: #to avoid expensive computation if not necessary
            mean = np.mean(np.array(self.selecting_dataframe(start_date_long, end_date_long, remove_weekends = remove_weekends)['Price']))
            self.long_term_vol = self.volatility(start_date_long, end_date_long, annualized=True, remove_weekends = remove_weekends)
            Brownian_motion = np.cumsum(np.random.randn(n))
            means = np.array([mean+self.long_term_vol*Brownian_motion[i] for i in range(n)])
        df = self.selecting_dataframe(end_date, end_date) #for start price of diffusion model
        start_price = float(df['Price'])
        self.dates = dates
        return volatilities, mean_reversions, means, n, start_price

    def pilipovic_fixed_forward(self, volatilities, mean_reversions, means, n, start_price):
        '''
        Inputs - arrays of volatilities, mean_reverisions, means, length of simulation and starting diffusion price
        Outputs - Diffusion model over simulation dates
        Numerically solves stochastic differential equation of the pilipovic process.
        Here is considered standard brownian motion at each time step. The considered time
        step is a day. The model is run on n time steps of a day each.
        The function takes into account switches between summer and winter in considered time period.
        If construction parameter forward_diffusion is False, then diffusion is around one scenario of long term mean
        with mean changing at each step according to long term volatility of the market (wiener process).
        '''
        Spot_curve = [start_price]
        Brownian_motion = np.random.randn(n) 
        if self.forward_diffusion: #diffusion autour de forward fixe
            for i in range(1, n):
                forward = self.forward_curve[i*4//n]  #will be more complicated if forward data is in a different format than webscraped one
                G_k = Spot_curve[-1]
                G_k1 = mean_reversions[i]*(forward - G_k) + volatilities[i]*Brownian_motion[i] + G_k  # N(0,1)
                Spot_curve.append(G_k1)
        else:                         #diffusion autour d'une moyenne accompagnée d'un mouvement determiné par la volatilité à long terme
            for i in range(1, n):
                 G_k = Spot_curve[-1]
                 G_k1 = mean_reversions[i]*(means[i] - G_k) + volatilities[i]*Brownian_motion[i] + G_k 
                 Spot_curve.append(G_k1)
        return Spot_curve


    def pilipovic_dynamic_forward_simple(self, start_date_long, end_date_long, start_date, end_date, end_date_sim):
        '''
        Input - dates, same format as pilipovic_fixed_parameters
        Output - Forward curve and a corresponding spot price with each forward.
        A dynamic forward curve is considered to generate different spot scenarios for each day.
        As the forward curve changes daily, we change the evolution of spot prices accordingly.
        Both forward and spot dynamics are governed by pilipovic processes. Historical data for
        both spot and forward is needed.
        We only consider time to maturity of one month as we are not computing full spot diff scenarios.
        Lack of historical data - no seasonality adjustment for webscraped datasets
        '''
        f_volatilities = self.forward_volatility(start_date, end_date)
        f_mean_reversions, f_means = self.forward_mean_reversion(start_date, end_date)
        s_volatilities, s_mean_reversion, s_means, n, start_price = self.pilipovic_fixed_parameters(start_date_long, end_date_long, start_date, end_date, end_date_sim, remove_weekends=True)
        Forward_curve = [self.fetch_forward(end_date)[0]]
        Spot_curve = [start_price]
        for i in range(1, n):
            F_k = Forward_curve[-1]
            F_k1 = f_mean_reversions[0]*(f_means[0] - F_k) + f_volatilities[0]*np.random.randn() + F_k
            Forward_curve.append(F_k1)
            G_k = Spot_curve[-1]
            G_k1 = s_mean_reversion[i]*(F_k - G_k) + s_volatilities[i]*np.random.randn() + G_k
            Spot_curve.append(G_k1)
        return Forward_curve, Spot_curve

    def pilipovic_dynamic_forward_multiple(self, start_date_long, end_date_long, start_date, end_date, end_date_sim, m = 1):
        '''
        See pilipovic_dynamic_forward_simple.
        Here we generate multiple spot scenarios for each daily forward curve
        Input - dates + number of spot scenarios per day
        Output - 4 forward curves of different times to maturity + Associated daily spot diffusions in dictionary. 
        '''
        f_volatilities = self.forward_volatility(start_date, end_date)
        f_mean_reversions, f_means = self.forward_mean_reversion(start_date, end_date)
        dates = self.daterange(end_date, end_date_sim, remove_weekends=True)
        n = len(dates)
        Forward_curve = [[self.fetch_forward(end_date)[i]] for i in range(4)]
        Spot_curve = {dates[0] : self.multiple_price_scenarios(start_date_long, end_date_long, start_date, end_date, end_date_sim, m)[0]}  #For first step it's a normal diffusion around fixed forward
        for i in range(1, n):
            for k in range(4):
                F = Forward_curve[k][-1]
                F1 = f_mean_reversions[k]*(f_means[k] - F) + f_volatilities[k]*np.random.randn() + F
                Forward_curve[k].append(F1)
            for _ in range(m):
                Spot = [Spot_curve[dates[i-1]][0][1]]  #We initialize all the diffusion model with the second step of first previous spot diffusion model 
                for y in range(1, n):
                    current_date = self.date_without_weekends(dates[i], n)  #skipping week-ends as forwards not issued
                    s_mean_reversion = self.mean_reversion_sum if (current_date.month in self.summer_months) else self.mean_reversion_win
                    s_volatility = self.vol_sum if (current_date.month in self.summer_months) else self.vol_win
                    G = Spot[-1]
                    G1 = s_mean_reversion*(Forward_curve[4*y//n][-1] - G) + s_volatility*np.random.randn() + G
                    Spot.append(G1)
                Spot_curve.setdefault(dates[i], []).append(Spot)
        return Forward_curve, Spot_curve


    def multiple_price_scenarios(self, start_date_long:str, end_date_long:str, start_date:str, end_date:str, end_date_sim:str, n:int):
        '''
        Generates n number of spot price scenarios using a pilipovic process for the evolution 
        dynamics of the spot price. Forward price is based on future curve at start_date.
        Date format is %Y-%m-%d. Gives table with all spot curves and a curve of mean calculated 
        over all spot curves.
        Input - dates as pilipovic_fixed_parameters. n integer for number of diffusion scenarios.
        Output - tab with n scenarios. Moyenne with mean of all diffusion scenarios. Mean depends
        on self.forward_diffusion i.e whether we are diffusing aroud forward. If True then mean is []
        else we are diffusing around historical mean in which case means represents this long term 
        historical mean on which a random walk with long term volatility was added.
        '''
        moyenne, tab = [], []
        volatilities, mean_reversions, means, p, start_price = self.pilipovic_fixed_parameters(start_date_long, end_date_long, start_date, end_date, end_date_sim)
        for _ in range(n):
            tab.append(self.pilipovic_fixed_forward(volatilities, mean_reversions, means, p, start_price))
        for i in range(len(tab[0])):
            moyenne.append(sum(tab[k][i] for k in range(len(tab)))/len(tab))
        return tab, moyenne, means, n

    def show_multiple(self, tab, moyenne, means, n):
        '''
        Function to display the multiple price scenarios created as well as the mean curve
        and forward prices associated.
        '''
        dates = self.dates
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        ax1.xaxis_date()
        fig.autofmt_xdate()
        ax1.set_title(f'{n} Spot price scenarios')
        ax1.set_ylabel("Spot Price (€/MWh)")
        for i in range(len(tab)):
            ax1.plot(dates, tab[i], lw=1)
        if self.forward_diffusion:
            curve = [self.forward_curve[i*4//len(tab[0])] for i in range(len(tab[0]))]
            ax2.plot(dates, curve, label='Prix forward', lw=2)
        else:
            curve = means
            ax2.plot(dates, curve, label='Long Term Mean with variation', lw=2)
        ax2.plot(dates, moyenne, lw=2, label='Mean of diffusion scenarios')
        ax2.set_ylabel('€/MWh')
        ax2.legend(loc='upper left')
        plt.show()


    def random_forest_dataset_spot(self, start_date, end_date, remove_weekends = False, test=False):
        '''
        Input - start and end date for train and test of the random_forest regression algorithm. Test
        is boolean, True for just testing on historical data, False for test AND training datasets.
        Output - train and test datasets ready to be used by a random forest regression algorithm. 
        Features are Date, means over the past 1 month, 2 months ... 6 months. Volatilities over 1, 2 ... 6 months.
        Mean reversion over 1, 2 ... 6 months.
        '''
        df = self.selecting_dataframe(self.previous_date(start_date, 0), self.previous_date(end_date, 0), remove_weekends = remove_weekends) #This has spot prices and dates 
        dates  = df['Day'].to_list()
        means, vols, mean_revs = [], [], []
        for element in dates:  #Getting all the necessary features for our daterange
            element_mean, element_vol, element_mean_rev = [], [], []
            for i in range(1, 7):
                prev_date = self.previous_date(element, i)
                vol, m = self.volatility(prev_date, element.strftime('%Y-%m-%d'), remove_weekends = remove_weekends, mean_include = True)  #Get volatility, mean and mean_reversion
                rev = self.mean_reversion(prev_date, element.strftime('%Y-%m-%d'), remove_weekends = remove_weekends) 
                element_vol.append(vol)
                element_mean.append(m)
                element_mean_rev.append(rev)
            means.append(element_mean)
            vols.append(element_vol)
            mean_revs.append(element_mean_rev)
        for k in range(1, 7):  #Adding new columns to dataframe
            df[f'Mean {k} months'] = np.array([means[p][k-1] for p in range(len(means))])
            df[f'Vols {k} months'] = np.array([vols[p][k-1] for p in range(len(vols))])
            df[f'Mean_revs {k} months'] = np.array([mean_revs[p][k-1] for p in range(len(mean_revs))])
        first_spot = self._dataset.loc[self._dataset['Day'] == self.previous_date(start_date, 0)]['Price']
        X = np.array(df.drop(columns = ['Price', 'Day']))
        y = np.array(df['Price'])
        if test:
            return X, y
        else:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=27) #shuffled
            return X_train, X_test, y_train, y_test

    def random_forest_regression_train_test(self, X_train, X_test, y_train, y_test, n_estimators:int):
        '''
        Input - test and training datasets. Number of estimators for random forest algorithm
        Output - the predicted values, RMSE and the regressor object
        '''
        regressor = RandomForestRegressor(n_estimators = n_estimators)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        RMSE = metrics.mean_squared_error(y_test, y_pred)**0.5
        return y_pred, RMSE, regressor

    def random_forest_regression_predict(self, start_date_sim, end_date_sim, regressor):
        '''
        Input - start and end date of simulation. Trained regressor object.
        Output - The prediction of our model over new dates 
        Each feature being calculated with a rolling window. 
        '''
        dates = self.daterange(start_date_sim, end_date_sim, remove_weekends = True) #No weekends in dataframe New_Power_Next_Spot
        for element in dates:
            pass
        pass



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
    #     self.vol_sum = self.volatility(start_date, end_date, summer=True)
    #     self.vol_win = self.volatility(start_date, end_date, winter=True)
    #     if self.vol_sum == 0:  #if we don't have enough data we might encounter a problem with volatility estimation
    #         self.vol_sum = self.vol_win
    #     elif self.vol_win == 0:
    #         self.vol_win = self.vol_sum
    #     self.mean_reversion_sum = self.mean_reversion(start_date, end_date, summer=True)
    #     self.mean_reversion_win = self.mean_reversion(start_date, end_date, summer=False)
    #     if self.mean_reversion_sum == 0:
    #         self.mean_reversion_sum = self.mean_reversion_win
    #     elif self.mean_reversion_win == 0:
    #         self.mean_reversion_sum = self.mean_reversion_sum
    #     forward_curve = self.fetch_forward(end_date)
    #     mean = forward_curve[int(simul_date.split('-')[1]) - int(end_date.split('-')[1])]
    #     if simul_date.split('-')[1] in self.summer_months:
    #         sigma = self.vol_sum
    #         alpha = self.mean_reversion_sum
    #     else:
    #         sigma = self.vol_win
    #         alpha = self.mean_reversion_win
    #     if not spot_price:
    #         G_0 = df['Price'].to_list()[-1]
    #     else:
    #         G_0 = spot_price
    #     return float(alpha*(mean - G_0) + sigma*np.random.randn() + G_0)