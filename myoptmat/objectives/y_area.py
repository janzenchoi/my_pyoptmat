"""
 Title:         The y_area objective function
 Description:   The objective function for calculating the vertical areas between two curves
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
from myoptmat.objectives.__objective__ import __Objective__

# The Y Area class
class Objective(__Objective__):
    
    # Returns the vertical area, assuming x values are the same
    def get_value(self, exp_curve:dict, prd_curve:dict) -> float:
        exp_y_avg = abs(np.average(exp_curve["y"]))
        y_area = [abs(prd_curve["y"][i] - exp_curve["y"][i]) for i in range(len(prd_curve["y"]))]
        return np.average(y_area) / exp_y_avg