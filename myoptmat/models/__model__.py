"""
 Title:         Model Template
 Description:   Contains the basic structure for a model class
 Author:        Janzen Choi

"""

# Libraries
import importlib.util, os, pathlib, sys, torch
import myoptmat.math.mapper as mapper
from typing import Callable
from pyoptmat import optimize
from pyoptmat.models import ModelIntegrator
from pyoptmat.temperature import ConstantParameter

# Constants
EXCLUSION_LIST = ["__model__", "__pycache__"]

# Model Template
class __Model__():
    
    # Constructor
    def __init__(self):
        self.name = None
        self.param_dict = {}
        self.param_mapper_dict = {}
        self.device = None
        self.opt_model = None
    
    # Prepares the model at the start - placeholder that must be overriden
    def prepare(self) -> None:
        raise NotImplemented("The 'setup' function has not been implemented!")
    
    # Returns the integrator of the model - placeholder that must be overridden
    def get_integrator(self, *param, **kwargs) -> None:
        raise NotImplemented("The 'define_model' function has not been implemented!")

    # Sets the name of the model
    def set_name(self, name:str) -> None:
        self.name = name

    # Gets the name of the model
    def get_name(self) -> str:
        return self.name

    # Adds parameter information
    def define_param(self, name, l_bound, u_bound):
        if name in self.param_dict.keys():
            raise ValueError(f"Parameter '{name}' has been defined multiple times!")
        self.param_dict[name] = {"l_bound": l_bound, "u_bound": u_bound}

    # Returns the dictionary of parameters
    def get_param_dict(self):
        return self.param_dict

    # Sets the device
    def set_device(self, device_type:str="cpu") -> None:
        self.device = torch.device(device_type)
    
    # Returns a constant value
    def get_constant(self, value:float) -> ConstantParameter:
        return ConstantParameter(torch.tensor(value, device=self.device))
    
    # # Returns a value object
    def get_param_object(self, value:float, l_bound:float=None, u_bound:float=None) -> ConstantParameter:
        scaling = optimize.bounded_scale_function((
            torch.tensor(l_bound, device=self.device),
            torch.tensor(u_bound, device=self.device),
        ))
        return ConstantParameter(value, scaling=scaling)
    
    # Custom clamping and unmapping function
    def clamp_and_unmap(self, param_mapper:mapper.Mapper) -> Callable:
        in_l_bound, in_u_bound = param_mapper.get_in_bounds()
        in_l_bound = torch.tensor(in_l_bound, device=self.device)
        in_u_bound = torch.tensor(in_u_bound, device=self.device)
        out_l_bound, out_u_bound = param_mapper.get_out_bounds()
        factor = (out_u_bound - out_l_bound) / (in_u_bound - in_l_bound)
        return lambda x: (x - out_l_bound) / factor + in_l_bound # no clamp
    
    # Define the parameter mappers
    def define_param_mapper_dict(self, param_mapper_dict:dict):
        self.param_mapper_dict = param_mapper_dict
    
    # Calibrates the model with a set of parameters
    def make_model(self, *params, **kwargs) -> ModelIntegrator:
        param_objects = []
        for i in range(len(params)):
            param_name = list(self.param_dict.keys())[i]
            param_mapper = self.param_mapper_dict[param_name]
            scaling = self.clamp_and_unmap(param_mapper)
            param_object = ConstantParameter(params[i], scaling=scaling)
            param_objects.append(param_object)
        integrator = self.get_integrator(*param_objects, **kwargs).to(self.device)
        return integrator
    
    # Gets the deterministic model for optimisation
    def get_opt_model(self, initial_values:list, block_size:int) -> optimize.DeterministicModel:
        return optimize.DeterministicModel(
            lambda *args,
            **kwargs: self.make_model(*args, block_size=block_size, **kwargs),
            self.param_dict.keys(),
            torch.tensor(initial_values, device=self.device)
        )

# Creates and return a model
def get_model(model_name:str, device_type:str="cpu") -> __Model__:

    # Get available models in current folder
    models_dir = pathlib.Path(__file__).parent.resolve()
    files = os.listdir(models_dir)
    files = [file.replace(".py", "") for file in files]
    files = [file for file in files if not file in EXCLUSION_LIST]
    
    # Raise error if model name not in available models
    if not model_name in files:
        raise NotImplementedError(f"The model '{model_name}' has not been implemented")

    # Prepare dynamic import
    module_path = f"{models_dir}/{model_name}.py"
    spec = importlib.util.spec_from_file_location("model_file", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Import and initialise model
    from model_file import Model
    model = Model()
    
    # Prepare model and return it
    model.set_name(model_name)
    model.set_device(device_type)
    model.prepare()
    return model