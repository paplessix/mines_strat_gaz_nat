from NewDiffusion import DiffusionSpot
import pandas as pd

path1 = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/spot_€_MWh_PEG.csv'
path2 = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/forward_€_MWh_PEG.csv'

#best to run from an anaconda prompt or terminal rather than from interactive console
diff = DiffusionSpot(path1, path2)
start_date_long, start_date = '2020-02-15', '2020-02-15'   #Due to non availability of data on a longer time scale.
end_date_long, end_date = '2020-03-30', '2020-03-30'
end_date_sim = '2020-07-30'
tab, moyenne, means, n = diff.multiple_price_scenarios(start_date_long, end_date_long, start_date, end_date, end_date_sim, 20)
diff.show_multiple(tab, moyenne, means, n)
#Pour écrire le array de diffusion de prix en csv
columns = diff.daterange(end_date, end_date_sim)
df = pd.DataFrame(data=tab, columns = columns)
df.head(10)
# diff.illustrating_mean_reversion(start_date, end_date) #optional to illustrate mean reversion, we see the parameter is quite close to 0.



# To try the simulation out around the historical mean and not a diffusion around forward price.
# diff = DiffusionSpot(path1, path2, forward_diffusion=False)
# diff.show_multiple(start_date_long, end_date_long, start_date, end_date, end_date_sim, 20)