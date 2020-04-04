import numpy as np 
import scipy 
import math
import pandas as import pd 

class Diffusion:

    def __init__(self, path, parameter):
        self._dataset = pd.read_csv(path)
