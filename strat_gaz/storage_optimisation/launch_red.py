import pandas as pd 
import numpy as np  
from pathlib import Path
from tree_reducer import Scenario_builder

path = Path(__file__).parent.parent / 'Data'/ 'Diffusion_model.csv'
df = pd.read_csv(path)
df  =df .transpose()
scen = Scenario_builder()
scen.data_loader_df(df)
scen.plot_tree()
scen.scenario_tree_construction(1)
scen.plot_tree()
# scen.nx_graph_builder()
# scen.plot_graph()
