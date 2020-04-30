from mines_strat_gaz_nat.storage_optimisation.tree_reducer import Scenario_builder
import numpy as np
import pandas as pd
from pathlib import Path
csv_path = Path(__file__).parent / "test.csv"
csv_path_duo = Path(__file__).parent / "test_duo.csv"

def test_builder() : 
    scen = Scenario_builder()
    try :   
        scen.tree
    except AttributeError:
        assert True
    else:
        assert False

def test_data_loader():
    scen = Scenario_builder()
    scen.data_loader(csv_path)
    try :   
        scen.tree
    except AttributeError:
        assert False
    else:
        assert True

# Symétrie du calcul de la distance 

def test_dist_sym():
    scen = Scenario_builder()
    scen.data_loader(csv_path_duo)
    assert scen.distance_scen_pair(0,1, scen.T) == scen.distance_scen_pair(1,0,scen.T)

def test_dist_T_max():
    scen = Scenario_builder()
    scen.data_loader(csv_path_duo)
    for i in range(scen.T):
        assert scen.distance_scen_pair(0,1,i)==i



