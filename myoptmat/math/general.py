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

# Prints a list of values with formatting and padding
def print_value_list(pre_text:str, value_list:list=[], padding:int=20, end="\n"):
    padding_str = (padding-len(pre_text)) * " "
    str_list = ["{:0.3}".format(float(value)) for value in value_list]
    str_str = f"[{', '.join(str_list)}]" if str_list != [] else ""
    print(f"{pre_text}{padding_str}{str_str}", end=end)