from NewDiffusion import DiffusionSpot
import pandas as pd
#best to run from an anaconda prompt or terminal rather than from interactive console

path1 = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/Power_next_spot.xlsx'
path2 = ''

#Initial formatting of these dataframes so the module can work correctly specifically for the PowerNext file
df = pd.read_excel(path1)
df = df[['Trading Day', 'Daily Average Price\n(DAP)']]
df.columns = ['Day', 'Price']
df = df.loc[df['Price'] != '-']
df.drop_duplicates(inplace=True, subset=['Day'])
df.reset_index(inplace=True, drop=True)
df.to_csv('C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/New_Power_Next_spot.csv')


#Now we can work with our newly formatted file!
path_new = 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/New_Power_Next_spot.csv'
diff = DiffusionSpot(path_new, path2, forward_diffusion=False)   #Diffusing around historical mean not a forward price
start_date_long = '2012-09-10'
end_date_long = '2013-09-10'
start_date = '2013-07-10'
end_date = '2013-09-10'
end_date_simul = '2013-12-10'
diff.multiple_price_scenarios(start_date_long, end_date_long, start_date, end_date, end_date_sim, 20)
# diff.show_multiple(start_date_long, end_date_long, start_date, end_date, end_date_sim, 20)



#Let's compare some results, i.e compare the mean of our diffusion model with the actual evolution of prices in that simulated time period
