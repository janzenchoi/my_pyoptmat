"""
 Title:         General
 Description:   For general math related helper functions
 Author:        Janzen Choi

"""

# Libraries
import math

# Returns a list of indexes corresponding to thinned data
def get_thin_indexes(src_data_size, dst_data_size):
    step_size = src_data_size/dst_data_size
    thin_indexes = [math.floor(step_size*i) for i in range(1,dst_data_size-1)]
    thin_indexes = [0] + thin_indexes + [src_data_size-1]
    return thin_indexes

# Clamps values to bounds
def clamp(value:float, l_bound:float, u_bound:float):
    if isinstance(value, list):
        return [clamp(v, l_bound, u_bound) for v in value]
    return min(max(value, l_bound), u_bound)
