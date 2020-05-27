from NewDiffusion import DiffusionSpot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path1 = 'C:/Users/spart/Documents/MinesParis/1A/Info/ProjetInfo/data_save_02_04/spot_€_MWh_PEG.csv'
path2 = 'C:/Users/spart/Documents/MinesParis/1A/Info/ProjetInfo/data_save_02_04/forward_€_MWh_PEG.csv'

#best to run from an anaconda prompt or terminal rather than from interactive console
diff = DiffusionSpot(path1, path2)
start_date_long, start_date = '2020-02-17', '2020-02-17'   #Due to non availability of data on a longer time scale.
end_date_long, end_date = '2020-03-30', '2020-03-30'
end_date_sim = '2020-07-30'

#Let's get our forward and spot scenarios
Forward_curve, Spot_curve = diff.pilipovic_dynamic_forward_simple(start_date_long, end_date_long, start_date, end_date, end_date_sim)

#To display them
dates = diff.daterange(end_date, end_date_sim, remove_weekends=True)
fig, ax = plt.subplots()
ax.xaxis_date()
fig.autofmt_xdate()
ax.plot(dates, Forward_curve, label = 'Forward_curve - Month+1')
ax.plot(dates, Spot_curve , label = 'Spot_curve ')
fig.legend()
plt.show()

#To write into csv file
rows = diff.daterange(end_date, end_date_sim, remove_weekends=True)
columns = ['Forward_curve', 'Spot_curve']
df = pd.DataFrame(data = np.column_stack((Forward_curve, Spot_curve)), columns = columns, index=rows)
df.to_csv('Diffusion_model_dynamic_forward')

#One forward diffusion and multiple spot diffusions
Forward_curve2, Spot_curve2 = diff.pilipovic_dynamic_forward_multiple(start_date_long, end_date_long, start_date, end_date, end_date_sim, m = 2)
fig2, ax2 = plt.subplots()
ax2.xaxis_date()
fig2.autofmt_xdate()
ax2.plot(dates, Forward_curve2[0], label = 'Forward_curve - Month+1')
ax2.plot(dates, Forward_curve2[1] , label = 'Forward_curve - Month+2')
ax2.plot(dates, Forward_curve2[2], label = 'Forward_curve - Month+3')
ax2.plot(dates, Forward_curve2[3] , label = 'Forward_curve - Month+4')
fig2.legend()
plt.show()

#Writing to csv