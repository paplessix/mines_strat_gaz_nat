import numpy as np 
import scipy 
import math
import pandas as import pd 

#Pour juste faire tourner le script sans devoir faire appel à des classes spécifiques
parser = argparse.ArgumentParser()
parser.add_argument("nom_fichier", type=str, help='')
args = parser.parse_args()
nom = args.nom_fichier


class DiffusionSpot:

    def __init__(self, path = nom, parameter):
        '''
        Initialise the Diffusion class. The historical data for the creation of the diffusion model
        of spot prices must be in a csv file, placed in the same repository as the script. If not,
        the path to the csv should be specified.
        '''
        self._dataset = pd.read_csv(path)
