from strat_gaz.storage_optimisation.tree_reducer import Scenario_builder
import numpy as np
import pandas as pd
from pathlib import Path
csv_path = Path(__file__).parent / "test.csv"
csv_path_duo = Path(__file__).parent / "test_duo.csv"

# Vérification de la non présence
# d'un tree avant l'initialisaton 


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
    scen.data_loader_csv(csv_path)
    try :   
        scen.tree
    except AttributeError:
        assert False
    else:
        assert True

# Symétrie du calcul de la distance 

def test_dist_sym():
    scen = Scenario_builder()
    scen.data_loader_csv(csv_path_duo)
    assert scen.distance_scen_pair(0,1, scen.T) == scen.distance_scen_pair(1,0,scen.T)

# Vérification de la somme

def test_dist_T_max():
    scen = Scenario_builder()
    scen.data_loader_csv(csv_path_duo)
    for i in range(scen.T):
        assert scen.distance_scen_pair(0,1,i)== i

# Vérification de la symétrie des matrices

def test_sym_mat_dist():
    scen = Scenario_builder()
    scen.data_loader_csv(csv_path)
    for i in range(scen.T):
        mat = scen.mat_distance_scen(i)
        assert scen.C_K_J.all() == scen.C_K_J.transpose().all() 






