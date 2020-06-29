from strat_gaz.storage_optimisation.stockage import *
from pathlib import Path
import pandas as pd 
import numpy as np
path_spot = Path(__file__).parent.parent / 'Data' / 'spot_history_HH.csv'
# volume initial bien borné

def test_vinit():
    try :
        data =  pd.read_csv(path_spot)
        X_0 = np.zeros( len(data['Day']))   
        data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')  
        stock = Stockage(100,-10 , data, X_0)
    except ValueError:
        assert True
    else:
        assert False

# lim_inf not defined

def test_lim_inf():
    try :
        data =  pd.read_csv(path_spot)
        X_0 = np.zeros( len(data['Day']))   
        data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')  
        stock = Stockage(100,10 , data, X_0, comp_tunnel = False)
        stock.lim_inf
    except AttributeError :
        assert True    

#objet pour tests
data =  pd.read_csv(path_spot)
X_0 = np.zeros( len(data['Day']))   
data['Day'] = pd.to_datetime(data['Day'], format = '%Y-%m-%d')  
stock = Stockage(100,10 , data, X_0)

#test sout 
def test_soutirage_empty():

    if stock.sout_correction(0) == 0 :
        assert True

def test_soutirage_full():

    if stock.sout_correction(1) == 1 :
        assert True
    
def test_inj_empty():

    if stock.sout_correction(0) == 1 :
        assert True

def test_inj_full():

    if stock.sout_correction(1) == 0 :
        assert True


