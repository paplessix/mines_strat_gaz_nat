""" Module to extract data from a graph on a website like powernext
"""
import sys
import time

import pandas as pd
from selenium import webdriver


def pos_data_extract(string_data: str):
    """Function that  process the string_data from the powernext website to extract
    the interesting coordinates values
    Parameters :
        - string_data : str, The website data
    Return :
        - list of tuple
    """
    data = list(map(float, map(test_quote, list(
        filter(lambda x: not("M" in x or "L" in x), string_data.split(' '))))))
    # decompilation des coordonnées 2 par 2
    coordinates_couple = [i for i in chunks(data, 2)]
    return coordinates_couple


def test_quote(string, carac='"'):
    """function that handle the data which contain '"' character
    Parameters:
        - string : str the string to handle
        - carac = '"' the character to find
    """
    if carac in string:
        return string[:-2]  # this is based on the data architecture
    return string


def chunks(lst: list, number):
    """Generator that yield successive n-sized chunks from lst.
    parameters :
        - lst : list of data processed
    Return :
        - Generator
    """
    for i in range(0, len(lst), number):
        yield lst[i:i + number]


def data_list_append(prices, price, dates, date, prediction_types, prediction_type):
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
    else:
        prices.append(price)
        prediction_types.append(prediction_type)
        dates.append(date)


class Browser:
    """ Class that launchs Powernext in chrome and that scrap the spot data from dynamic graph
    """
    def __init__(self):
        self.delay = 10
        self.gnl_types = None
        self.bloc = None
        self.driver = None

    def chrome_launcher(self, link):
        """Function that launch a chrome browser
        """
        # Create a new instance of Chrome
        option = webdriver.ChromeOptions()
        option.add_argument("--start-maximized") # start fullscreen
        print(' Loading url', link)
        # Optional argument, if not specified will search path.
        self.driver = webdriver.Chrome(
            'chromedriver.exe', chrome_options=option)
        self.driver.get(link)

    def gnl_finder(self, bloc_number=0):
        """
        Find a list of all GNL market types present on the website, thy're represented by the corresponding WebElement.
        And add it as an attribute to the class
        parameters :
            - self
            - bloc_number : position of the graph in the DOM
        Return :
            - None
        Exemple : [PEG, ZEE, ETH, ...]
        """
        blocs = self.driver.find_elements_by_css_selector(
            "div.standard-page-block.standard-page-body")
        # 0 is the position of the interesting DOM <div>
        self.bloc = blocs[bloc_number]
        self.gnl_types = self.bloc.find_elements_by_tag_name("li")

    def scraper_iterator(self, specific_type=False,
                         link="https://www.powernext.com/spot-market-data", bloc_number=0):
        """Function that determine all the GNL prices for all market on Powernext
        This function is an iterator on all the market types.
        On each graph it is basically a function that move a cursor from left to right with
        a fixed step until the mouse reach
        the right limit of the graph. The step in px is fixed to be slightly under the
        effective distance between 2 points of the graph.
        Parameters :
            - self : a Browser object
            - specific_type : default = False, else list containig the GNL market to consider
            - link : the weblink to use  we recommand not to change it unless being aware of
            the DOM structure
        Return a generator :
            - title : str ( maybe useless)
            - info : str ( type of price and unit change this ? )
            - active : str ( the type of GNL)
            - data : pd.DataFrame ()

        """
        self.chrome_launcher(link)  # Launch Chrome

        time.sleep(2)  # Let the user actually see something!
        self.gnl_finder(bloc_number)
        for gnl_type in self.gnl_types:
            if (not specific_type) or (gnl_type.text in specific_type):
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(
                    gnl_type).perform()  # click on the button
                webdriver.ActionChains(self.driver).double_click(
                    gnl_type).perform()  # click another time to make it work
                # Let the time for the website to load the graph. For more explanation see doc.
                time.sleep(5)
                active = self.bloc.find_element_by_class_name(
                    'active').text  # the currently considered market
                # find the graph in the DOM
                chart_container = self.bloc.find_elements_by_class_name(
                    'highcharts-graph')[bloc_number]
                string_data = chart_container.value_of_css_property("d")
                chart_container = self.bloc.find_elements_by_class_name(
                    'highcharts-graph')[bloc_number]
                # Put the mouse in the middle of the graph
                webdriver.ActionChains(self.driver).move_to_element(
                    chart_container).perform()
                # Extract data
                pos_data = pos_data_extract(string_data)
                # the max_min px position of the graph on the website
                min_x = min(pos_data, key=lambda x: x[0])[0]
                max_x = max(pos_data, key=lambda x: x[0])[0]
                min_y = min(pos_data, key=lambda x: x[1])[1]
                max_y = max(pos_data, key=lambda x: x[1])[1]
                # Determine coordinate of the middle of the box
                pos_abs_middle_x = (max_x - min_x)/2 + chart_container.rect['x'] + min_x
                pos_abs_middle_y = (max_y - min_y)/2 + chart_container.rect['y'] + min_y
                # Move to the left limit of the graph
                offset_x = pos_data[0][0] + chart_container.rect['x'] - pos_abs_middle_x
                offset_y = pos_data[0][1] + chart_container.rect['y'] - pos_abs_middle_y
                webdriver.ActionChains(self.driver).move_by_offset(
                    offset_x, offset_y).perform()
                active_pos_x = pos_data[0][0]

                dates, prediction_types, prices = [], [], []
                last_date = None
                step = 10  # efficient step for the size of the screen

                while active_pos_x <= max_x-2:
                    # Extract considered date
                    label = self.driver.find_element_by_class_name(
                        "highcharts-label")
                    texte = label.find_element_by_tag_name("text")
                    texte = texte.find_elements_by_tag_name("tspan")
                    prediction_type, date = texte[0].text.split(' ')

                    if date == last_date:
                        pass # mean that the step forward didn't get on the next point
                        # so we do nothing new
                    else:
                        # Extract price
                        info_price = texte[1].text.split(' ')
                        unit = info_price[1]
                        price = float(info_price[0])
                        last_date = date
                        data_list_append(prices, price, dates,
                                         date, prediction_types, prediction_type)
                    # Move one step forward
                    active_pos_x = active_pos_x + step
                    webdriver.ActionChains(
                        self.driver).move_by_offset(step, 0).perform()
                # Gather the collected features
                table = pd.DataFrame(
                    {'Trading Day': dates, 'Prediction Type': prediction_types, 'Price': prices})
                yield unit, active, table
            else:
                # Mean that the process should not extract data on the currently
                # considered GNL market
                print('not now')
        self.driver.quit()


def main():
    """Function that iter the iterator
    """
    browser = Browser()
    for i in browser.scraper_iterator():
        print(i)

if __name__ == '__main__':
    sys.exit(main())
