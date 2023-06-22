import sys; sys.path += [".."]
from myoptmat.api import API
api = API()
api.define_device("cpu")
api.define_model("test")
# api.set_initial_values(0.5, 0.5, 0.5, 0.5, 0.5)
# api.read_data("tensile/AirBase_20_D5.csv")
api.read_data("test_tensile.csv")
# api.set_data_scale(True)
api.set_param_scale(True)
api.optimise()