import sys; sys.path += [".."]
from myoptmat.api import API

api = API()
api.define_device("cpu")
api.define_model("test")

# api.read_file("tensile/AirBase_20_D5.csv")
api.read_folder("test")

api.initialise_param("n",   0)
api.initialise_param("eta", 0)
api.initialise_param("s0",  0)
api.initialise_param("R",   0)
api.initialise_param("d",   0)

api.scale_param("n",   0, 1)
api.scale_param("eta", 0, 1)
api.scale_param("s0",  0, 10)
api.scale_param("R",   0, 10)
api.scale_param("d",   0, 10)

api.scale_data("time",        0, 1)
api.scale_data("strain",      0, 1)
api.scale_data("stress",      0, 1)
api.scale_data("temperature", 0, 1)
api.scale_data("cycle",       0, 1)

api.optimise(block_size=5, iterations=20, display=True)
api.display_results()