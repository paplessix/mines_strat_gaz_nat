

from selenium import webdriver 
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import sys
import time


def table_scraper(link = 'https://www.powernext.com/futures-market-data'):
    """
    Function that returns an iterator on the table of forward prices of different GNL Types
    Input :
        - URL : str
    Output
    """
    # Create a new instance of Chrome
    option = webdriver.ChromeOptions()
    option.add_argument("— incognito")

    driver = webdriver.Chrome('./chromedriver.exe')  # Optional argument, if not specified will search path.
    driver.get(link)

    #Search table blocs
    blocs  = driver.find_elements_by_css_selector("div.standard-page-block.standard-page-body")
    bloc_month = blocs[-3]
    bloc_season = blocs[-2] # TO DO : Implement  a fonction to distinguish wich div are relevant to scrap
    GNL_types_month= bloc_month.find_elements_by_tag_name ("li") # Find all possible GNL types
    GNL_types_season= bloc_season.find_elements_by_tag_name ("li") # Find all possible GNL types
    
    for type_month, type_season in zip( GNL_types_month, GNL_types_season) : # iterate on this types
        
        
        # Reinitialize var
        table_month, table_season,table = None, None, None
        no_month, no_season = False, False

        webdriver.ActionChains(driver).double_click(type_month).perform()# click on the button
        time.sleep(1)
        webdriver.ActionChains(driver).double_click(type_season).perform() 
        time.sleep(3) # TO DO : Wait for the table to be loaded
        #find_title and infos
        info = bloc_month.find_element_by_class_name('data-table-title').text
        
        active_month = bloc_month.find_element_by_class_name('active').text
        active_season = bloc_season.find_element_by_class_name('active').text
        print(active_season,active_month)
        
        if active_month != active_season :
            raise TypeError


        # find table div
        try:
            div_month = bloc_month.find_element_by_class_name('table-responsive')
        except NoSuchElementException:
            no_month = True
        else:
            # find html
            html_month  = div_month.get_attribute('innerHTML')
            table_month=pd.read_html(str(html_month))[0] # convert to DataFrame
        
        try:
            div_season = bloc_season.find_element_by_class_name('table-responsive')
        except NoSuchElementException:
            no_season = True
            
        else:
            # find html
            html_season  = div_season.get_attribute('innerHTML')
            table_season=pd.read_html(str(html_season))[0] # convert to DataFrame
         
        if no_month :
            table = table_season
        elif  no_season :
            table = table_month
        else :
            table = pd.merge(table_month, table_season)
        yield info, active_month, table

    driver.quit()

def main():
    for i in table_scraper():
        print(i)

if __name__ == '__main__':
    sys.exit(main())





