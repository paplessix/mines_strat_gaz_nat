

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import sys
import time

class Browser : 
    def __init__(self):
        self.delay = 10

    def chrome_launcher(self,link):
        # Create a new instance of Chrome
        option = webdriver.ChromeOptions()
        option.add_argument("— incognito")
        print(link)
        self.driver = webdriver.Chrome('./chromedriver.exe')  # Optional argument, if not specified will search path.
        self.driver.get(link)
    
    def wait_is_loaded(self, bloc,path_direction, type_of_path):
        path = {"css_selector" : By.CSS_SELECTOR, "class_name" : By.CLASS_NAME} 
        try:
            WebDriverWait(bloc, self.delay).until(EC.visibility_of_element_located((path[type_of_path],path_direction)))
        except TimeoutException:
            print( 'data_exception')
    
    def GNL_finder(self):
            blocs  = self.driver.find_elements_by_css_selector("div.standard-page-block.standard-page-body")
            self.bloc_month = blocs[-3]
            self.bloc_season = blocs[-2] # TO DO : Implement  a fonction to distinguish wich div are relevant to scrap
            self.GNL_types_month= self.bloc_month.find_elements_by_tag_name ("li") # Find all possible GNL types
            self.GNL_types_season= self.bloc_season.find_elements_by_tag_name ("li") # Find all possible GNL types
            self.GNL_types_text = [tile.text for tile in self.GNL_types_month]
            print(self.GNL_types_text)

    def table_scraper(self):
        no_month, no_season = False, False
        table_month, table_season,table = None, None, None
        #find_title and infos        
        info = self.bloc_month.find_element_by_class_name('data-table-title').text
        active_month = self.bloc_month.find_element_by_class_name('active').text
        active_season = self.bloc_season.find_element_by_class_name('active').text
        
        
        if active_month != active_season :
            raise TypeError


        # find table div
        try:
            div_month = self.bloc_month.find_element_by_class_name('table-responsive')

        except NoSuchElementException:
            no_month = True
        else:
            # find html
            html_month  = div_month.get_attribute('innerHTML')
            table_month=pd.read_html(str(html_month))[0] # convert to DataFrame
        
        try:
            div_season = self.bloc_season.find_element_by_class_name('table-responsive')
        except NoSuchElementException:
            no_season = True
            
        else:
            # find html
            html_season  = div_season.get_attribute('innerHTML')
            table_season=pd.read_html(str(html_season))[0] # convert to DataFrame    

        return table_month, table_season, no_month, no_season, info, active_month


    def scraper_iterator(self,specific_type = False, link = 'https://www.powernext.com/futures-market-data' ):
        """
        Function that returns an iterator on the table of forward prices of different GNL Types
        Input :
            - URL : str
        Output
        """
        self.chrome_launcher(link)
        self.GNL_finder()

        for type_month, type_season in zip( self.GNL_types_month, self.GNL_types_season) : # iterate on this types
            if  (not specific_type)  or (type_season.text in specific_type):        
                # Change GNL types 
                webdriver.ActionChains(self.driver).double_click(type_season).perform()
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(type_month).perform()# click on the button
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(type_season).perform()# chelou ça amrche sur mon ordi
                #Wait for everything to be loaded 
                time.sleep(3) # Best Way to ensure that the table is perfectly loaded for now
                
                table_month, table_season, no_month, no_season, info, active = self.table_scraper()
                
                if no_month :
                    table = table_season
                    print('no_month')
                elif  no_season :
                    table = table_month
                    print('no_season')
                else :
                    table = pd.merge(table_month, table_season)
                    print("merge")
                yield info, active, table
            else:
                print('not now')

        self.driver.quit()

def main():
    browser = Browser()
    for i in browser.scraper_iterator(specific_type="GPL"):
        print(i)

if __name__ == '__main__':
    sys.exit(main())





