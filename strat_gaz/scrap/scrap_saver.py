"""
Module that control the scrapping, and ensure the interface between the
user commands and the scraping. See Doc for commands.
"""

import argparse
import os
import sys

import pandas as pd

from scrap.graph_scraper import Browser as Browser_spot
from scrap.table_scraper import Browser as Browser_forward


def filename_constructor(directory, info, active, price_type):
    """ Function that constructs the path to the file depending on the GNL market type
    Return:
        - a string corresponding to the normalized filename, hadling the specificities
        of the set of data.
    """
    active = active.replace(' ', '_') # Create a more database-friendly name
    unit = info.split(' ')[-1].replace('/', '_') # Get rid of / and second date of the WE
    filename = directory + '/' + price_type + '_' + unit + '_'+active+'.csv'
    # build the complete name
    return filename


def list_csv_dir(directory : str):
    """
    Sparse all the csv files present in the active directory
    Parameters :
        - directory : str, the working directory
    Return :
        - a list of all the csv files in the directory
    """
    files = list(map(lambda x: directory + '/' + x,
                     filter(lambda x: x[-4:] == '.csv', os.listdir(directory))))
    return files


def data_initializer(directory, price_type, specific_type=False):
    """Sparse as much data as possible supposing that there is no one existing before
    Create the data base architecture
    """
    functions = {"forward": Browser_forward, "spot": Browser_spot}
    browser = functions[price_type]() # select spot or forward browser
    # Run the Iterator
    for info, active, table in browser.scraper_iterator(specific_type):
        table['Trading Day'] = pd.to_datetime(
            table['Trading Day'], format='%Y-%m-%d')
        table = table.sort_values('Trading Day') #sort dates increasingly
        filename = filename_constructor(directory, info, active, price_type)
        table.to_csv(filename, index=False) #save data


def data_updater(directory, price_type, specific_type=False):
    """Read the daily data and add to the existing DB only the missing one
        If the data do not exist, it will create a table, create the files
        at the directory variable location. Updates spot or forward prices,
        and can update only a specific market_type. 
        
        Parameters :
            - directory : str, the place where to write/search the files
            - price_type : str, the price type needed to be updated (spot or forward)
            - specific_type : str, the desired market type, ex [ PEG, ZEE, ETH]
        Return : 
            - None
    """
    functions = {"forward": Browser_forward, "spot": Browser_spot}
    if price_type in functions.keys() :
        browser = functions[price_type]()
    else :
        raise KeyError # use spot or forward

    list_csv = list_csv_dir(directory) # Get the current existing db
    print(list_csv)
    for info, active, table in browser.scraper_iterator(specific_type):
        table['Trading Day'] = pd.to_datetime(
            table['Trading Day'], format='%Y-%m-%d')
        table = table.sort_values('Trading Day')
        filename = filename_constructor(directory, info, active, price_type)
        if filename in list_csv: # if the market already exists
            existing_data = pd.read_csv(filename)
            existing_data = existing_data.sort_values('Trading Day') # be sure it is sorted
            existing_data['Trading Day'] = pd.to_datetime(
                existing_data['Trading Day'], format='%Y-%m-%d')
            last_date = max(existing_data['Trading Day']) # get last recording
            print(' last_recorded_date', last_date)
            table = table[table['Trading Day'] > last_date] # Get new data
            existing_data = existing_data.append(table, sort=False) # Add the new data
            existing_data.to_csv(filename, index=False) # save the new db
        else:
            table.to_csv(filename, index=False) # create the DB


def main():
    """
    Function that understand the command line pass by the user in the terminal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="where to update files")
    parser.add_argument(
        "-p", "--product", help=" Product type required default = ['spot','forward]")
    parser.add_argument("-s", "--specific", help="Specific market desired")
    args = parser.parse_args()

    if not args.directory:
        directory = './strat_gaz/scrap/last_save'
    else:
        directory = args.directory

    if args.product == 'spot':
        data_updater(directory, 'spot', args.specific)
    elif args.product == 'forward':
        data_updater(directory, 'forward', args.specific)
    else:
        data_updater('./scrap/last_save', 'forward', args.specific)
        data_updater('./scrap/last_save', 'spot', args. specific)


if __name__ == '__main__':
    sys.exit(main())
