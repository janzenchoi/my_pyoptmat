"""
 Title:         MyOptMat API
 Description:   API for interacting with the MyOptMat script
 Author:        Janzen Choi

"""

# Libraries
import os, time
import myoptmat.optimisation.optimiser as optimiser
import warnings; warnings.filterwarnings("ignore")
import torch; torch.set_default_tensor_type(torch.DoubleTensor)

# API Class
class API:
    
    # Constructor
    def __init__(self, name="", input_dir:str="./data", output_dir:str="./results"):
        
        # Define internal pathing
        time_str = time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
        folder_epilogue = f"_{name}" if name != "" else ""
        self.__input_path__ = input_dir
        self.__output_path__ = f"{output_dir}/{time_str}{folder_epilogue}"
        
        # Internal variables
        self.__csv_file_list__ = []
        self.__model_name__ = "undefined"
        self.__device_type__ = "cpu"
        self.__initial_values__ = None
        self.__scale_params__ = False
        self.__scale_data__ = False
        
        # Create output folders if they don't exist
        os.mkdir(output_dir) if not os.path.exists(output_dir) else None
        os.mkdir(self.__output_path__) if not os.path.exists(self.__output_path__) else None
        
    # Reads data from a folder of CSV files
    def read_folder(self, csv_folder:str) -> None:
        csv_folder_path = f"{self.__input_path__}/{csv_folder}"
        csv_file_path_list = [f"{csv_folder_path}/{file}" for file in os.listdir(csv_folder_path) if file.endswith(".csv")]
        self.__csv_file_list__ += csv_file_path_list
        
    # Reads data from a CSV file
    def read_file(self, csv_file:str) -> None:
        csv_file_path = f"{self.__input_path__}/{csv_file}"
        self.__csv_file_list__.append(csv_file_path)
        
    # Defines the device
    def define_device(self, device_type:str="cpu") -> None:
        self.__device_type__ = device_type
    
    # Defines the model
    def define_model(self, model_name:str) -> None:
        self.__model_name__ = model_name
    
    # Defines the initial values
    def set_initial_values(self, *initial_values:list) -> None:
        self.__initial_values__ = initial_values
    
    # Turns on scaling of parameters to [0,1]
    def set_param_scale(self) -> None:
        self.__scale_params__ = True
    
    # Turns on scaling of data to [0,1]
    def set_data_scale(self) -> None:
        self.__scale_data__ = True
    
    # Define scales for the model parameters
    def define_param_scales(self, bounds=dict) -> None:
        self.__param_scale_list__ = bounds
    
    # Define scales for the model parameters
    def define_data_scales(self, bounds=dict) -> None:
        self.__data_scale_list__ = bounds
    
    # Initiates optimisation
    def optimise(self, iterations:int=5, block_size:int=40) -> None:
        opt = optimiser.Optimiser(self.__model_name__, self.__scale_params__, self.__scale_data__)
        opt.initialise_params(self.__initial_values__)
        opt.initialise_data(self.__csv_file_list__)
        opt.initialise_settings(block_size)
        opt.get_gradient()
        opt.conduct_optimisation(iterations)
        opt.display_results()