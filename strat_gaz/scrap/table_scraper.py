"""
Module that scraps data from forward prices on powernext
"""
import sys
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException


class Browser:
    """ Class that launchs Powernext in chrome and that scrap the spot data from dynamic graph
    """

    def __init__(self):
        self.delay = 10
        self.gnl_types_month = None
        self.gnl_types_season = None
        self.bloc_month = None
        self.bloc_season = None
        self.driver = None

    def chrome_launcher(self, link: str):
        """Function that launch a chrome browser
        Parameter:
            - The url
        """
        # Create a new instance of Chrome
        option = webdriver.ChromeOptions()
        option.add_argument("— incognito")
        print(link)
        # Optional argument, if not specified will search path.
        self.driver = webdriver.Chrome('./chromedriver.exe')
        self.driver.get(link)

    def gnl_finder(self):
        """
        Find a list of all GNL market types present on the website,
        they're represented by the corresponding WebElement.
        And add it as an attribute to the class
        parameters :
            - self

        Return :
            - None
        Exemple : [PEG, ZEE, ETH, ...]
        """
        blocs = self.driver.find_elements_by_css_selector(
            "div.standard-page-block.standard-page-body")
        self.bloc_month = blocs[-4]  # Depending on the DOM architecture
        self.bloc_season = blocs[-3]  # Dependinf on the DOM architecture
        self.gnl_types_month = self.bloc_month.find_elements_by_tag_name(
            "li")  # Find all possible gnl types
        self.gnl_types_season = self.bloc_season.find_elements_by_tag_name(
            "li")  # Find all possible gnl types

    def table_scraper(self):
        """ Function that extract the data and all the relevant informations
        from the html code of a loaded page
        Return :
            - table_month
            - table_season
            - no_month
            - no_season
            - active_month
        """
        no_month, no_season = False, False
        table_month, table_season = None, None
        #find_title and infos
        info = self.bloc_month.find_element_by_class_name(
            'data-table-title').text
        active_month = self.bloc_month.find_element_by_class_name(
            'active').text
        active_season = self.bloc_season.find_element_by_class_name(
            'active').text

        if active_month != active_season:
            raise TypeError

        # find table div
        try:
            div_month = self.bloc_month.find_element_by_class_name(
                'table-responsive')

        except NoSuchElementException:
            no_month = True
        else:
            # find html
            html_month = div_month.get_attribute('innerHTML')
            table_month = pd.read_html(str(html_month))[
                0]  # convert to DataFrame

        try:
            div_season = self.bloc_season.find_element_by_class_name(
                'table-responsive')
        except NoSuchElementException:
            no_season = True

        else:
            # find html
            html_season = div_season.get_attribute('innerHTML')
            table_season = pd.read_html(str(html_season))[
                0]  # convert to DataFrame

        return table_month, table_season, no_month, no_season, info, active_month

    def scraper_iterator(self, specific_type=False,
                         link='https://www.powernext.com/futures-market-data'):
        """
        Function that returns an iterator on the table of forward prices of different
        gnl market Types
        Input :
            - URL : str
            - specific_type : the type of gnl market to consider, default --> all
            - link : tje url website where to find the data ,
            please be careful to find a website with a similar DOM
        Output :
            - info
            - active
            - table
        """
        self.chrome_launcher(link)
        self.gnl_finder()
        # iterate on the gnl market types
        for type_month, type_season in zip(
                self.gnl_types_month, self.gnl_types_season):

            if (not specific_type) or (type_season.text in specific_type):
                # Change gnl types
                webdriver.ActionChains(self.driver).double_click(
                    type_season).perform()
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(
                    type_month).perform()  # click on the button
                time.sleep(1)
                webdriver.ActionChains(self.driver).double_click(
                    type_season).perform()  # Click a second time to be sure it works
                # Wait for everything to be loaded
                # Best Way to ensure that the table is perfectly loaded for now
                time.sleep(3)

                table_month, table_season, no_month, no_season, info, active = self.table_scraper()

                if no_month:
                    table = table_season
                    print('no_month')
                elif no_season:
                    table = table_month
                    print('no_season')
                else:
                    table = pd.merge(table_month, table_season)
                    print("merge")
                yield info, active, table
            else:
                # if the currently active gnl_market doesn't need to be
                # scrapped
                print('not now')
                # we do nothing

        self.driver.quit()


def main():
    """
    go through iterator default mode
    """
    browser = Browser()
    for i in browser.scraper_iterator(specific_type="GPL"):
        print(i)


if __name__ == '__main__':
    sys.exit(main())
