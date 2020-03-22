from table_scraper import table_scraper
import datetime
import csv
import pandas as pd
def filename_constructor(directory,info,active):
    active = active.replace(' ','_')
    unit = info.split(' ')[-1].replace('/', '_')
    filename = directory + '/' + 'forward_' + unit +'_'+active+'.csv'
    return filename


def data_initializer(directory, price_type = "forward"):
    if price_type == "forward":
        for title, info, active, table in table_scraper():
            filename = filename_constructor(directory, info, active)
            table.to_csv(filename, index = False)

# data_initializer('./Web_Scraping')


def data_updater(directory):
    for title, info, active, table in table_scraper(): 
        filename = filename_constructor(directory, info, active)
        existing_data = pd.read_csv(filename)
        existing_data['Trading Day'] = pd.to_datetime(existing_data['Trading Day'],format = '%Y-%m-%d')
        table['Trading Day'] = pd.to_datetime(table['Trading Day'],format = '%Y-%m-%d')
        print(max(existing_data['Trading Day']))
        print(max(existing_data['Trading Day']))

data_updater('rr')