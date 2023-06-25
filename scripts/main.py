import sys; sys.path += [".."]
from myoptmat.api import API
api = API(output_here=True)

api.define_device("cpu")
api.define_model("test")

# api.read_file("tensile/AirBase_20_D5.csv")
api.read_file("test/test_1.csv")
# api.read_folder("test")

api.scale_param("n",   0, 10)
api.scale_param("eta", 0, 10)
api.scale_param("s0",  0, 1)
api.scale_param("R",   0, 1)
api.scale_param("d",   0, 1)

api.scale_data("time",        0, 1)
api.scale_data("strain",      0, 1)
api.scale_data("stress",      0, 1)
api.scale_data("temperature", 0, 1)
api.scale_data("cycle",       0, 1)

api.record(iterations=10)
api.optimise(iterations=10)
