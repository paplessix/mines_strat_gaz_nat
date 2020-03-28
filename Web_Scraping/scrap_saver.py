from table_scraper import table_scraper
from graph_scraper import graph_scraper
import datetime
import csv
import pandas as pd
import sys



def filename_constructor(directory,info,active, price_type):
    """ Function that construct the path to the file depending on the GNL type
    """
    active = active.replace(' ','_')
    unit = info.split(' ')[-1].replace('/', '_')
    filename = directory + '/' + price_type + '_' + unit +'_'+active+'.csv'
    return filename


def data_initializer(directory, price_type ):
    """Sparse as much data as possible supposing that there is no one existing before
    Create the data base architecture
    """
    if price_type == "forward":
        for info, active, table in table_scraper():
            filename = filename_constructor(directory, info, active, price_type)
            table = table.sort_values('Trading Day')
            table.to_csv(filename, index = False)

    if price_type == "spot":
        for unit, active, table in graph_scraper():
            filename = filename_constructor(directory, unit, active, price_type)
            table = table.sort_values('Trading Day')
            table.to_csv(filename, index = False)




def data_updater(directory, price_type ):
    """Read the daily data and add to the existing DB only the missing one 
    """
    if price_type == "forward":
        for info, active, table in table_scraper(): 
            filename = filename_constructor(directory, info, active,price_type)
            existing_data = pd.read_csv(filename)
            existing_data = existing_data.sort_values('Trading Day')
            existing_data['Trading Day'] = pd.to_datetime(existing_data['Trading Day'],format = '%Y-%m-%d')
            table['Trading Day'] = pd.to_datetime(table['Trading Day'],format = '%Y-%m-%d')
            last_date  = max(existing_data['Trading Day'])

            table = table[ table['Trading Day'] > last_date ]
            table = table.sort_values('Trading Day')

            existing_data = existing_data.append(table, sort = False)
            existing_data.to_csv(filename, index = False) 
    elif price_type == 'spot':
        
        for unit, active, table in graph_scraper():

            filename = filename_constructor(directory, unit, active,price_type)
            existing_data = pd.read_csv(filename)
            existing_data = existing_data.sort_values('Trading Day')
            existing_data['Trading Day'] = pd.to_datetime(existing_data['Trading Day'],format = '%Y-%m-%d')
            table['Trading Day'] = pd.to_datetime(table['Trading Day'],format = '%Y-%m-%d')
            last_date  = max(existing_data['Trading Day'])
            table = table[ table['Trading Day'] > last_date ]
            table = table.sort_values('Trading Day')

            existing_data = existing_data.append(table, sort = False)
            existing_data.to_csv(filename, index = False)


def main():
    data_initializer('./Web_Scraping','spot')
    #data_updater('./Web_Scraping','forward')
    #data_updater('./Web_Scraping','spot')

if __name__ == '__main__':
    sys.exit(main())