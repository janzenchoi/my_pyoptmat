"""
 Title:         Objective Template
 Description:   Contains the basic structure for an objective class
 Author:        Janzen Choi

"""

# Libraries
import importlib.util, os, pathlib, sys

# Constants
EXCLUSION_LIST = ["__objective__", "__pycache__"]

# Objective Template
class __Objective__():
    
    # Constructor
    def __init__(self):
        self.name = None

    # Sets the name of the objective
    def set_name(self, name:str) -> None:
        self.name = name

    # Gets the name of the objective
    def get_name(self) -> str:
        return self.name
    
    # Returns an error value
    def get_value(self, exp_curve:dict, prd_curve:dict) -> float:
        raise NotImplementedError("The 'get_value' function has not been implemented!")

# Creates and return an objective
def get_objective(objective_name:str) -> __Objective__:

    # Get available objectives in current folder
    objectives_dir = pathlib.Path(__file__).parent.resolve()
    files = os.listdir(objectives_dir)
    files = [file.replace(".py", "") for file in files]
    files = [file for file in files if not file in EXCLUSION_LIST]
    
    # Raise error if objective name not in available objectives
    if not objective_name in files:
        raise NotImplementedError(f"The objective '{objective_name}' has not been implemented")

    # Prepare dynamic import
    module_path = f"{objectives_dir}/{objective_name}.py"
    spec = importlib.util.spec_from_file_location("objective_file", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Import and initialise objective
    from objective_file import Objective
    objective = Objective()
    
    # Prepare objective and return it
    objective.set_name(objective_name)
    return objective