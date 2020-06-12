from NewDiffusion import DiffusionSpot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#best to run from an anaconda prompt or terminal rather than from interactive console

#t0 = time.time()
path1 = 'Power_next_spot.xlsx'
path2 = ''

#Initial formatting of these dataframes so the module can work correctly specifically for the PowerNext file
df = pd.read_excel(path1)
df = df[['Trading Day', 'Daily Average Price\n(DAP)']]
df.columns = ['Day', 'Price']
df = df.loc[df['Price'] != '-']
df.drop_duplicates(inplace=True, subset=['Day'])
df.reset_index(inplace=True, drop=True)
df.to_csv('C:/Users/spart/Documents/MinesParis/1A/Info/ProjetInfo/New_Power_Next_spot.csv', date_format = '%Y-%m-%d', columns=['Day', 'Price'], index=False)


#Now we can work with our newly oformatted file!
path_new = 'C:/Users/spart/Documents/MinesParis/1A/Info/ProjetInfo/New_Power_Next_spot.csv'
diff = DiffusionSpot(path_new, path2, forward_diffusion=False)

#Getting the new dataset full of rich features which we will use in a random forest regression algorithm
#Train and test will be done over 4 years
start_date = '2010-07-12'
end_date = '2014-07-10'
X_train, X_test, y_train, y_test = diff.random_forest_dataset_spot(start_date, end_date)
y_pred, RMSE, regressor = diff.random_forest_regression_train_test(X_train, X_test, y_train, y_test, 100)
print(f'RMSE:{RMSE}') 
print(f'Feature importances {regressor.feature_importances_}')

#Now we will use the trained and tested model to predict another time series
start_date_sim = '2014-07-11'
end_date_sim = '2014-09-11'
X_other_test, y_other_test = diff.random_forest_dataset_spot(start_date_sim, end_date_sim, test=True)
y_other_pred = regressor.predict(X_other_test)
