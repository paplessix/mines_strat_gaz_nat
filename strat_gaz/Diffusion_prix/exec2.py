from NewDiffusion import DiffusionSpot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#best to run from an anaconda prompt or terminal rather than from interactive console

t0 = time.time()
path1 = '../Data/Power_next_spot.xlsx'
path2 = ''

#Initial formatting of these dataframes so the module can work correctly specifically for the PowerNext file
df = pd.read_excel(path1)
df = df[['Trading Day', 'Daily Average Price\n(DAP)']]
df.columns = ['Day', 'Price']
df = df.loc[df['Price'] != '-']
df.drop_duplicates(inplace=True, subset=['Day'])
df.reset_index(inplace=True, drop=True)
df.to_csv('../Data/New_Power_Next_spot.csv', date_format = '%Y-%m-%d', columns=['Day', 'Price'], index=False)


#Now we can work with our newly formatted file!
path_new = '../Data/New_Power_Next_spot.csv'
diff = DiffusionSpot(path_new, path2, forward_diffusion=False)   #Diffusing around historical mean not a forward price
start_date_long = '2011-09-8'
end_date_long = '2013-09-10'
start_date = '2012-09-10'
end_date = '2013-09-10'
end_date_sim = '2014-09-10'
number_of_diffusion = 1000
tab, moyenne, long_term_means, n = diff.multiple_price_scenarios(start_date_long, end_date_long, start_date, end_date, end_date_sim, number_of_diffusion)
diff.show_multiple(tab, moyenne, long_term_means, n)

#Let's examine all our parameters
print(f'mean_reversion_sum = {diff.mean_reversion_sum}, mean_reversion_win =  {diff.mean_reversion_win}, vol_sum = {diff.vol_sum}, vol_win = {diff.vol_win}, long_term_vol = {diff.long_term_vol}')


#To compare with actual values
df2 = diff.selecting_dataframe(end_date, end_date_sim)
dates = diff.daterange(end_date, end_date_sim)
fig, ax = plt.subplots()
ax.xaxis_date()
fig.autofmt_xdate()
ax.plot(np.array(df2['Day'])[:-1], np.array(df2['Price'])[:-1], label = 'Real price')
ax.plot(dates, tab[0], label = 'One diffusion scenario')
ax.legend(loc='upper left')
plt.show()

print(f'Script took {time.time() - t0}s to run')


#To write diffusion model into csv file
final_tab = tab + [moyenne]
columns = diff.daterange(end_date, end_date_sim)
rows = [f'simulation n°{i}' for i in range(n)]
rows.append('moyenne scenarios')
df = pd.DataFrame(data = final_tab, columns = columns, index = rows)
df.to_csv('../Data/Diffusion/Diffusion_model_historical_mean')


