from NewDiffusion import DiffusionSpot

diff = DiffusionSpot('C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/spot_€_MWh_PEG.csv', 'C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/data_save_02_04/forward_€_MWh_PEG.csv')
start_date = '2020-02-15'
end_date = '2020-03-30'
end_date_sim = '2020-07-30'
diff.multiple_price_scenarios(start_date, end_date, end_date_sim, 20)