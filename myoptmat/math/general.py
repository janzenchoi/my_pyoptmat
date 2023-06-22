"""
 Title:         General
 Description:   For general math related helper functions
 Author:        Janzen Choi

"""

# Clamps values to bounds
def clamp(value, l_bound, u_bound):
    if isinstance(value, list):
        return [clamp(v, l_bound, u_bound) for v in value]
    return min(max(value, l_bound), u_bound)