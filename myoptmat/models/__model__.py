"""
 Title:         Model Template
 Description:   Contains the basic structure for a model class
 Author:        Janzen Choi

"""

# Libraries
import importlib.util, os, pathlib, sys, torch
from pyoptmat import optimize
from pyoptmat.models import ModelIntegrator
from pyoptmat.temperature import ConstantParameter
import myoptmat.math.mapper as mapper

# Constants
PATH_TO_MODELS = "myoptmat/models"
EXCLUSION_LIST = ["__model__", "__pycache__"]

# Model Template
class __Model__():
    
    # Constructor
    def __init__(self):
        self.name = None
        self.param_list = []
        self.device = None
        self.opt_model = None
        self.scale_params = False
    
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
        param_names = self.get_param_names()
        if name in param_names:
            raise ValueError(f"Parameter '{name}' has been defined multiple times!")
        self.param_list.append({
            "name": name,
            "l_bound": l_bound,
            "u_bound": u_bound,
        })
    
    # Returns the parameter names
    def get_param_names(self):
        return [param["name"] for param in self.param_list]

    # Returns the list of parameters
    def get_param_list(self):
        return self.param_list

    # Sets the device
    def set_device(self, device_type:str="cpu") -> None:
        self.device = torch.device(device_type)
    
    # Returns a constant value
    def get_constant(self, value:float) -> ConstantParameter:
        return ConstantParameter(torch.tensor(value, device=self.device))
    
    # Returns a value object
    def get_param_object(self, value:float, l_bound:float, u_bound:float) -> ConstantParameter:
        return ConstantParameter(
            value,
            scaling=optimize.bounded_scale_function((
                torch.tensor(l_bound, device=self.device),
                torch.tensor(u_bound, device=self.device),
            ))
        )
    
    # Turns on scaling for parameters
    def set_param_scale(self) -> None:
        self.scale_params = True
    
    # Calibrates the model with a set of parameters
    def make_model(self, *params, **kwargs) -> ModelIntegrator:
        param_objects = []
        for i in range(len(params)):
            l_bound = 0 if self.scale_params else self.param_list[i]["l_bound"]
            u_bound = 1 if self.scale_params else self.param_list[i]["u_bound"]
            param_object = self.get_param_object(params[i], l_bound, u_bound)
            param_objects.append(param_object)
        integrator = self.get_integrator(*param_objects, **kwargs).to(self.device)
        return integrator
    
    # Gets the deterministic model for optimisation
    def get_opt_model(self, initial_values:list=(), block_size:int=1) -> optimize.DeterministicModel:
        param_names = self.get_param_names()
        initial_value_tensors = torch.tensor(initial_values, device=self.device)
        return optimize.DeterministicModel(
            lambda *args,
            **kwargs: self.make_model(*args, block_size=block_size, **kwargs),
            param_names,
            initial_value_tensors
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