from selenium import webdriver 
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import datetime
import sys
import pandas as pd

def pos_data_extract(string_data :str):

    data = list(map(float,map (test_quote ,list(filter(lambda x : not("M" in x or "L" in x ), string_data.split(' '))))))
    coordinates_couple = [ i for i in chunks(data,2)]
    return coordinates_couple

def test_quote( string , carac ='"' ):
    if carac in string:
        return string[:-2]
    else : 
        return string

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def data_list_append(prices, price, dates, date,prediction_types, prediction_type ):
    """ Function to help add new data to the list of data
    Handle the WE doublon
    """
    if '/' in date:
        prices.append(price)
        prediction_types.append(prediction_type)
        dates.append(date[:-3])
        prices.append(price)
        prediction_types.append(prediction_type)
        dates.append(date[:-5]+date[-2:])  
    else : 
        prices.append(price)
        prediction_types.append(prediction_type)
        dates.append(date)  

class Browser :
    def __init__(self):
        self.delay = 10

    def chrome_launcher(self,link):
        # Create a new instance of Chrome
        option = webdriver.ChromeOptions()

        #option.add_argument("--window-size=1920,1080")
        option.add_argument("--start-maximized")
        print(' Loading url',link)
        self.driver = webdriver.Chrome('chromedriver.exe',chrome_options=option)  # Optional argument, if not specified will search path.
        self.driver.get(link)

    
    def GNL_finder(self):
        """
        Find a list of all the successive GNL types present on the website.
        """
        blocs  = self.driver.find_elements_by_css_selector("div.standard-page-block.standard-page-body")
        self.bloc = blocs[0]
        self.GNL_types = self.bloc.find_elements_by_tag_name ("li")
        self.GNL_types_text = [tile.text for tile in self.GNL_types]
        print(self.GNL_types_text)


    def scraper_iterator(self,specific_type = False, link ="https://www.powernext.com/spot-market-data", bloc_number = 0 ):
        """Function that determine the last price of GNL in a graph
        Still need to do multiple type of GNL and more than one day (probleme des WE)
        For now just catch he last value but could catch all the values 
        In order to unify with table_scraper.py
        Return : 
            - title : str ( maybe useless)
            - info : str ( type of price and unit change this ? )
            - active : str ( the type of GNL)
            - data : pd.DataFrame ()

        """ 
        self.chrome_launcher(link)           
        time.sleep(2) # Let the user actually see something!
        
        self.GNL_finder()
        for GNL_type in self.GNL_types :
            if  (not specific_type)  or (GNL_type.text in specific_type):
                
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(GNL_type).perform()# click on the button
                webdriver.ActionChains(self.driver).double_click(GNL_type).perform()# On double clique pou être sur que ça marche
                time.sleep(5)

                active = self.bloc.find_element_by_class_name('active').text

                # find the graph

                chart_container = self.bloc.find_elements_by_class_name('highcharts-graph')[bloc_number]
                string_data = chart_container.value_of_css_property("d")
                chart_container = self.bloc.find_elements_by_class_name('highcharts-graph')[bloc_number]
                webdriver.ActionChains(self.driver).move_to_element(chart_container).perform()
                pos_data = pos_data_extract(string_data)
                ####
                init_pos_x = pos_data[0][0]
                init_pos_y = pos_data[0][1]
                min_x = min(pos_data, key= lambda x : x[0])[0]
                max_x = max(pos_data, key= lambda x : x[0])[0]
                min_y = min(pos_data, key= lambda x : x[1])[1]
                max_y = max(pos_data, key= lambda x : x[1])[1]
                # Height and Width of the box
                width = max_x - min_x
                height = max_y - min_y
                # Determine coordinate of the middle of the box
                origin_x = chart_container.rect['x']  
                origin_y = chart_container.rect['y']
                pos_abs_middle_x = width/2 + origin_x + min_x
                pos_abs_middle_y = height/2 + origin_y + min_y
              
                active_pos_x = pos_abs_middle_x
                active_pos_y = pos_abs_middle_y

                offset_x = init_pos_x + origin_x - active_pos_x
                offset_y = init_pos_y + origin_y - active_pos_y
                webdriver.ActionChains(self.driver).move_by_offset(offset_x,offset_y).perform()
                active_pos_x = init_pos_x
                active_pos_y = init_pos_y

                dates, prediction_types, prices  = [],[],[]
                last_date = None
                step = 10
                while active_pos_x <= max_x-2   :                   
                    label = self.driver.find_element_by_class_name("highcharts-label")
                    texte = label.find_element_by_tag_name("text")
                    texte = texte.find_elements_by_tag_name("tspan")
                    #print(texte[0].text.split(' '))
                    prediction_type, date = texte[0].text.split(' ')
                    if date == last_date :
                        pass
                    else :     
                        info_price = texte[1].text.split(' ')
                        unit = info_price[1]
                        price = float(info_price[0])
                        last_date = date
                        data_list_append(prices, price, dates, date, prediction_types, prediction_type)
                    active_pos_x = active_pos_x + step
                    webdriver.ActionChains(self.driver).move_by_offset(step,0).perform()
                table = pd.DataFrame({'Trading Day': dates,'Prediction Type': prediction_types,'Price' : prices}) 
                yield unit, active, table
            else:
                print('not now')
        self.driver.quit()

 
def main():
    browser = Browser()
    for i in browser.scraper_iterator():
        print(i)

if __name__ == '__main__':
    sys.exit(main())
