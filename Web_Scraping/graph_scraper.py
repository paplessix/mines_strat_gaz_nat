from selenium import webdriver 
from selenium.common.exceptions import TimeoutException
import time
import datetime
import sys
import pandas as pd


def test_quote( string , carac ='"' ):
    if carac in string:
        return string[:-2]
    else : 
        return string
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def pos_data_extract(string_data :str):

    data = list(map(float,map (test_quote ,list(filter(lambda x : not("M" in x or "L" in x ), string_data.split(' '))))))
    coordinates_couple = [ i for i in chunks(data,2)]
    print(coordinates_couple)
    return coordinates_couple

def data_list_append(prices, price, dates, date,prediction_types, prediction_type ):
    """ Function to help add new data to the list of data
    Handle the WE doublon
    """git 
    if '/' in date:
        
        if len(prediction_types) and prediction_types[-1] == 'WE':

            prices.append(price)
            prediction_types.append(prediction_type)
            dates.append(date[:-5]+date[-2:])    
        else :     
            prices.append(price)
            prediction_types.append(prediction_type)
            dates.append(date[:-3])
    else : 
        prices.append(price)
        prediction_types.append(prediction_type)
        dates.append(date)

def graph_scraper(link ="https://www.powernext.com/spot-market-data"  ):
    """Function that determine the last price of GNL in a graph
    Still need to do multiple type of GNL and more than one day (probleme des WE )
    For now just catch he last value but could catch all the values 
    In order to unify with table_scraper.py
    Return : 
        - title : str ( maybe useless)
        - info : str ( type of price and unit change this ? )
        - active : str ( the type of GNL)
        - data : pd.DataFrame ()

    """

    # Find yesterday's date 
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    
    # Create a new instance of Chrome
    option = webdriver.ChromeOptions()
    option.add_argument("— incognito")

    driver = webdriver.Chrome('chromedriver.exe')  # Optional argument, if not specified will search path.
    driver.get(link)
    
    #time.sleep(2) # Let the user actually see something!
    
    
    blocs  = driver.find_elements_by_css_selector("div.standard-page-block.standard-page-body")
    bloc = blocs[0]
    GNL_types = bloc.find_elements_by_tag_name ("li")
    for GNL_type in GNL_types :

        webdriver.ActionChains(driver).double_click(GNL_type).perform()# click on the button
        time.sleep(1)

        title = bloc.find_element_by_class_name('thecontent').text
        active = bloc.find_element_by_class_name('active').text

        #find the graph
        chart_container = bloc.find_elements_by_class_name('highcharts-graph')[0]

        string_data = chart_container.value_of_css_property("d")
        pos_data = pos_data_extract(string_data)


        ####
        
        
        min_x = min(pos_data, key= lambda x : x[0])[0]
        max_x = max(pos_data, key= lambda x : x[0])[0]


        min_y = min(pos_data, key= lambda x : x[1])[1]
        max_y = max(pos_data, key= lambda x : x[1])[1]

        # Height and Width of the box
        width = max_x - min_x
        height = max_y - min_y

        

        # Determine coordinate of the middle of the box
        pos_rel_middle_x = width/2 + min_x
        pos_rel_middle_y = height/2 + min_y

        webdriver.ActionChains(driver).move_to_element(chart_container).perform()
        active_pos_x = pos_rel_middle_x
        active_pos_y = pos_rel_middle_y
        dates, prediction_types, prices  = [],[],[]
        for new_pos_x,new_pos_y in pos_data:
                
            #Calculate the offset of the mousz move to touch the last point 
            offset_x = new_pos_x-active_pos_x
            offset_y = new_pos_y-active_pos_y

            webdriver.ActionChains(driver).move_by_offset(offset_x,offset_y).perform()

            active_pos_x =new_pos_x
            active_pos_y = new_pos_y       
            
            label = driver.find_element_by_class_name("highcharts-label")
            texte = label.find_element_by_tag_name("text")
            texte = texte.find_elements_by_tag_name("tspan")
            print(texte[0].text.split(' '))
            prediction_type, date = texte[0].text.split(' ')
            info_price = texte[1].text.split(' ')
            unit = info_price[1]
            price = float(info_price[0])

            data_list_append(prices, price, dates, date, prediction_types, prediction_type)
        table = pd.DataFrame({'Trading Day': dates,'Prediction Type': prediction_types,'Price' : prices}) 
        print(active, table)
    
    
    
    driver.quit()
        # time.sleep(6) # Let the user actually see something!

        

    
    
def main():
    graph_scraper()
if __name__ == '__main__':
    sys.exit(main())
