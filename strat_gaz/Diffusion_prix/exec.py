from NewDiffusion import DiffusionSpot
import pandas as pd
import numpy as np
import argparse 

path1 = '../scrap/last_save/spot_€_MWh_PEG.csv'
path2 = '../scrap/last_save/forward_€_MWh_PEG.csv'

#best to run from an anaconda prompt or terminal rather than from interactive console
diff = DiffusionSpot(path1, path2, forward_diffusion=True)
start_date_long, start_date = '', '2020-03-15'#Due to nonavailability of data on longer time scale.
end_date_long, end_date = '', '2020-05-28'
end_date_sim = '2020-08-28'  #DO NOT make simulation longer than 4 months

number_of_scenarios = 1000
tab, moyenne, means, n = diff.multiple_price_scenarios(start_date_long,
                                                       end_date_long, start_date,
                                                       end_date, end_date_sim, number_of_scenarios)
diff.show_multiple(tab, moyenne, means, n)

#To write diffusion model into csv file
final_tab = tab + [moyenne]
columns = diff.daterange(end_date, end_date_sim)
rows = [f'simulation n°{i}' for i in range(n)]
rows.append('moyenne scenarios')
df = pd.DataFrame(data=final_tab, columns = columns, index=rows)
df.to_csv('../Data/Diffusion/Diffusion_model_fixed_forward')
# diff.illustrating_mean_reversion(start_date, end_date) 
#optional to illustrate mean reversion, we see the parameter is quite close to 0.


