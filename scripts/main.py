import sys; sys.path += [".."]
from myoptmat.api import API

api = API()
api.define_device("cpu")
api.define_model("test")
# api.set_initial_values(0.5, 0.5, 0.5, 0.5, 0.5)
# api.read_file("tensile/AirBase_20_D5.csv")
api.read_folder("test")
api.set_data_scale()
# api.set_param_scale()
api.optimise(5, 20)