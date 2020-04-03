import pandas as pd
import numpy as np
import sys


def structurator( filename = './storage_optimisation/spot_history_HH.xls' ):
    conversion_btu = 0.294
    data = pd.read_excel(filename, sheet_name = 'Data 1', header = 2)
    data.columns = ['Day','Price']
    data['Price'] = data['Price']*0.294
    data.to_csv('spot_history_HH.csv', index = False)

if __name__ == '__main__':
    filename = './storage_optimisation/spot_history_HH.xls'
    sys.exit(structurator(filename))