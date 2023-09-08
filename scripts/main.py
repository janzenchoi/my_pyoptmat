import sys; sys.path += [".."]
from myoptmat.api import API

api = API(output_here=True)
api.define_device("cpu")
api.def_model("evp")
api.define_algorithm("adam")

api.read_file("tensile/AirBase_20_D5.csv")

api.scale_param("n",   0, 100)
api.scale_param("eta", 0, 10)
api.scale_param("s0",  0, 1)
api.scale_param("R",   0, 1)
api.scale_param("d",   0, 1)

api.scale_data("time",        0, 1)
api.scale_data("strain",      0, 1)
api.scale_data("stress",      0, 1)
api.scale_data("temperature", 0, 1)
api.scale_data("cycle",       0, 1)

api.start_opt(iterations=20000, record=1000, display=10)

