from NewDiffusion import DiffusionSpot
import pandas as pd
#best to run from an anaconda prompt or terminal rather than from interactive console

path_spot = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/PowernextGasSpot_DailyPricesVolumes.csv'
path_forward = ''

#Initial formatting of these dataframes so the module can work correctly
df = pd.read_csv(path_spot)
df.head()
