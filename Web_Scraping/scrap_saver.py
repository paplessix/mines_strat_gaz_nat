from table_scraper import Browser as Browser_forward
from graph_scraper import graph_scraper as Browser_spot # to change
import datetime
import csv
import pandas as pd
import sys
import os 



def filename_constructor(directory,info,active, price_type):
    """ Function that construct the path to the file depending on the GNL type
    """
    active = active.replace(' ','_')
    unit = info.split(' ')[-1].replace('/', '_')
    filename = directory + '/' + price_type + '_' + unit +'_'+active+'.csv'
    return filename

def list_csv_dir( directory):
    files = list(filter(lambda x : x[-4:]=='.csv',os.listdir(directory)))
    return files 


def data_initializer(directory, price_type):
    """Sparse as much data as possible supposing that there is no one existing before
    Create the data base architecture
    """
    functions= {"forward": Browser_forward, "spot" : Browser_spot }
    browser = functions[price_type]()

    for i in browser.scraper_iterator():
        filename = filename_constructor(directory, info, active, price_type)
        table = table.sort_values('Trading Day')
        table.to_csv(filename, index = False)

def data_updater(directory, price_type, specific_type = False ):
    """Read the daily data and add to the existing DB only the missing one 
    """
    functions= {"forward": Browser_forward, "spot" : Browser_spot }
    browser = functions[price_type]()
    list_csv = list_csv_dir(directory)
    for info, active, table in browser.scraper_iterator( specific_type): 
        table['Trading Day'] = pd.to_datetime(table['Trading Day'],format = '%Y-%m-%d')
        table = table.sort_values('Trading Day')
        filename = filename_constructor(directory, info, active,price_type)
        if filename in list_csv:
            existing_data = pd.read_csv(filename)
            existing_data = existing_data.sort_values('Trading Day')
            existing_data['Trading Day'] = pd.to_datetime(existing_data['Trading Day'],format = '%Y-%m-%d')
            last_date  = max(existing_data['Trading Day'])
            table['Trading Day'] = pd.to_datetime(table['Trading Day'],format = '%Y-%m-%d')
            table = table[ table['Trading Day'] > last_date ]
            table = table.sort_values('Trading Day')
            existing_data = existing_data.append(table, sort = False)
            existing_data.to_csv(filename, index = False)
        else : 
            table.to_csv(filename, index = False) 
        
def main():
    data_initializer('./Web_Scraping','forward')

if __name__ == '__main__':
    sys.exit(main())