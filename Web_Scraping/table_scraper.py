

from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd


def table_scraper(link = 'https://www.powernext.com/futures-market-data' : str, bloc_number = -2 : int):
    """
    Function that returns an iterator on the table of forward prices of different GNL Types
    Input :
        - URL : str
    Output
    """
    # Create a new instance of Chrome
    option = webdriver.ChromeOptions()
    option.add_argument("— incognito")

    driver = webdriver.Chrome('chromedriver.exe')  # Optional argument, if not specified will search path.
    driver.get(link)

    #Search table blocs
    blocs  = driver.find_elements_by_css_selector("div.standard-page-block.standard-page-body")
    bloc = blocs[bloc_number] # TO DO : Implement to distinguish wich div are relevant to test 
    types = bloc.find_elements_by_tag_name ("li") # Find all possible GNL types
    for type in types : # iterate on this types
        webdriver.ActionChains(driver).double_click(type).perform()# click on the button 
        #find_title and infos
        title = bloc.find_element_by_class_name('thecontent').text
        info = bloc.find_element_by_class_name('data-table-title').text
        active = bloc.find_element_by_class_name('active').text

        # find table div
        try:
            div = bloc.find_element_by_class_name('table-responsive')
        except NoSuchElementException:
            pass # If no data skip for today 
        else:
            # find html
            html  = div.get_attribute('innerHTML')
            table=pd.read_html(str(html))[0] # convert to DataFrame
            yield title, info, active, table

    driver.quit()

def main():
    for i in table_scraper():
        print(i)






