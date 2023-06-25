"""
 Title:         Mapper
 Description:   For linear mapping
 Author:        Janzen Choi

"""

# Libraries
import numpy as np

# Mapper class
class Mapper:
    
    # Constructor
    def __init__(self, in_l_bound:float=0, in_u_bound:float=1, out_l_bound:float=0, out_u_bound:float=1):
        self.in_l_bound = in_l_bound
        self.in_u_bound = in_u_bound
        self.out_l_bound = out_l_bound
        self.out_u_bound = out_u_bound
        self.distinct = in_l_bound == in_u_bound or out_l_bound == out_u_bound
    
    # Linearly maps a value (works for lists)
    def map(self, value:float) -> float:
        if self.distinct:
            return value
        if isinstance(value, list):
            return [self.map(v) for v in value]
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.in_l_bound) * factor + self.out_l_bound

    # Linearly unmaps a value (works for lists)
    def unmap(self, value:float) -> float:
        if self.distinct:
            return value
        if isinstance(value, list):
            return [self.unmap(v) for v in value]
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.out_l_bound) / factor + self.in_l_bound
    
    # Produces a random value between the out bounds
    def random(self) -> float:
        return np.random.uniform(self.out_l_bound, self.out_u_bound)

    # Returns the in bounds as a tuple
    def get_in_bounds(self) -> tuple:
        return self.in_l_bound, self.in_u_bound

    # Returns the out bounds as a tuple
    def get_out_bounds(self) -> tuple:
        return self.out_l_bound, self.out_u_bound