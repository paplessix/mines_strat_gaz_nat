import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import timedelta, date, datetime

path1 = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/spot_€_MWh_PEG.csv'
path2 = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/forward_€_MWh_PEG.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df1.columns = ['Day', 'Prediction', 'Price'] 
df1['Day'] = pd.to_datetime(df1['Day'])
forward_prediction = df2[['Trading Day', 'Month+1']]
forward_prediction['Trading Day'] = pd.to_datetime(df2['Trading Day'])
fig, ax = plt.subplots()
ax.xaxis_date()
fig.autofmt_xdate()
spot_on_forward = df1.loc[df1['Day'] >= date(2020, 3, 17)]
ax.plot(spot_on_forward['Day'], forward_prediction['Month+1'][:16])
ax.plot(spot_on_forward['Day'], spot_on_forward['Price'])