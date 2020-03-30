from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
import time
import datetime


def graph_scraper(link ="https://www.powernext.com/futures-market-data"  ):
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

    driver = webdriver.Chrome('C:/Users/spart/Documents/MinesParis/Info/ProjetInfo/githubprojet/mines_strat_gaz_nat/chromedriver.exe')  # Optional argument, if not specified will search path.
    driver.get(link)
    
    #time.sleep(2) # Let the user actually see something!

    #find the graph
    web_object = driver.find_element_by_class_name('highcharts-graph')

    pos_data = web_object.value_of_css_property("d")
    pos_data = str(pos_data)
    pos_data = pos_data.split(' ')


    ####
    # A terme remplacer ici par le Min et le Max dees valeurs de x et de y pour avoir la vrai valeur de la hauteur 


    pos_data_init = pos_data[1:3]
    pos_data_init[0]= float(pos_data_init[0])
    pos_data_init[1]= float(pos_data_init[1])

    #### Fin des trucs à changer 

    # last position 
    pos_data_of_the_day = pos_data[-2:]
    pos_data_of_the_day[0]= float(pos_data_of_the_day[0])
    pos_data_of_the_day[1]= float(pos_data_of_the_day[1][:-2])

    # Height and Width of the box
    width = pos_data_of_the_day[0] - pos_data_init[0]
    height = pos_data_of_the_day[1] - pos_data_init[1]



    chart_container = driver.find_elements_by_class_name('highcharts-graph')[0]

    # width  = chart_container.value_of_css_property("width")
    # height= chart_container.value_of_css_property("height")
    # print(width,height)
    # Determine coordinate of the middle of the box
    pos_rel_middle_x = width/2 + pos_data_init[0]
    pos_rel_middle_y = height/2 + pos_data_init[1]
    #Calculate the offset of the mouse move to touch the last point 
    offset_x = pos_data_of_the_day[0]-pos_rel_middle_x
    offset_y = pos_data_of_the_day[1]-pos_rel_middle_y

    webdriver.ActionChains(driver).move_to_element(chart_container).perform()
    webdriver.ActionChains(driver).move_by_offset(offset_x,offset_y).perform()


    label = driver.find_element_by_class_name("highcharts-label")
    texte = label.find_element_by_tag_name("text")
    prix = texte.find_elements_by_tag_name("tspan")[1]
    prix = prix.text.split(' ')
    unit = prix[1]
    prix = float(prix[0])
    driver.quit()
    # time.sleep(6) # Let the user actually see something!

    return (yesterday, prix, unit)
    
    

