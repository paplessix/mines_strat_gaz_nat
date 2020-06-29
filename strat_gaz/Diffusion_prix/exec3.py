from NewDiffusion import DiffusionSpot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path1 = '../scrap/last_save/spot_€_MWh_PEG.csv'
path2 = '../scrap/last_save/forward_€_MWh_PEG.csv'

diff = DiffusionSpot(path1, path2)
start_date_long, start_date = '', '2020-02-17'   #Due to non availability of data on a longer time scale.
end_date_long, end_date = '', '2020-06-11'
end_date_sim = '2021-06-11'

#Let's get our forward and spot scenarios
Forward_curve, Spot_curve = [], []
number_of_scenarios = 1000
for i in range(number_of_scenarios):
    F, S = diff.pilipovic_dynamic_forward_simple(start_date_long, end_date_long, start_date, end_date, end_date_sim)
    Forward_curve.append(F)
    Spot_curve.append(S)

#To display them
dates = diff.daterange(end_date, end_date_sim, remove_weekends=True)
fig, ax = plt.subplots()
ax.xaxis_date()
fig.autofmt_xdate()
ax.plot(dates, Forward_curve[0], label = 'Forward_curve - Month+1')
ax.plot(dates, Spot_curve[0], label = 'Spot_curve ')
ax.set_ylabel('€/MWh')
ax.set_title('Forward diffusion')
fig.legend(loc='upper right')
plt.show()

#To write into csv file
columns = diff.daterange(end_date, end_date_sim, remove_weekends=True)
rows = [f'simulation n°{i}' for i in range(number_of_scenarios)]
df = pd.DataFrame(data=Spot_curve, columns = columns, index=rows)
df.to_csv('../Data/Diffusion/Diffusion_model_dynamic_forward_1000')

#One forward diffusion and multiple spot diffusions
Forward_curve2, Spot_curve2 = diff.pilipovic_dynamic_forward_multiple(start_date_long, end_date_long, start_date, end_date, end_date_sim, m = 2)
fig2, ax2 = plt.subplots()
ax2.xaxis_date()
fig2.autofmt_xdate()
ax2.plot(dates, Forward_curve2[0], label = 'Forward_curve - Month+1')
ax2.plot(dates, Forward_curve2[1] , label = 'Forward_curve - Month+2')
ax2.plot(dates, Forward_curve2[2], label = 'Forward_curve - Month+3')
ax2.plot(dates, Forward_curve2[3] , label = 'Forward_curve - Month+4')
ax2.set_ylabel('€/MWh')
fig2.legend()
plt.show()