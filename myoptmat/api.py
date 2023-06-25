"""
 Title:         MycntMat API
 Description:   API for interacting with the MycntMat script
 Author:        Janzen Choi

"""

# Libraries
import os, time
import myoptmat.controller as controller
import warnings; warnings.filterwarnings("ignore")
import torch; torch.set_default_tensor_type(torch.DoubleTensor)

# API Class
class API:
    
    # Constructor
    def __init__(self, name="", input_dir:str="./data", output_dir:str="./results", output_here:bool=False):
        
        # Define internal pathing
        time_str = time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
        folder_epilogue = f"_{name}" if name != "" else ""
        self.__input_path__ = input_dir
        self.__output_path__ = f"{output_dir}/{time_str}{folder_epilogue}" if not output_here else "."

        # Define parameter variables
        self.__param_scale_dict__ = {}
        self.__initial_param_dict__ = {}

        # Define data variables
        self.__data_scale_dict__ = {}
        self.__csv_file_list__ = []
        
        # Define other internal variables
        self.__model_name__ = None
        self.__device_type__ = "cpu"
        self.__ctrl__ = controller.Controller()
        
        # Create output folders if they don't exist
        if not output_here:
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
    
    # Sets the initial value for a parameter
    def initialise_param(self, param_name:str, param_value:float) -> None:
        self.__initial_param_dict__[param_name] = float(param_value)
    
    # Sets the scale for a parameter
    def scale_param(self, param_name:str, l_bound:float=0, u_bound:float=1) -> None:
        self.__param_scale_dict__[param_name] = {"l_bound": l_bound, "u_bound": u_bound}
    
    # Sets the scale for a data header
    def scale_data(self, data_name:str, l_bound:float=0, u_bound:float=1) -> None:
        self.__data_scale_dict__[data_name] = {"l_bound": l_bound, "u_bound": u_bound}
    
    # Initialises the recorder
    def record(self, iterations:int=5) -> None:
        record_path = f"{self.__output_path__}/record"
        self.__ctrl__.initialise_recorder(record_path, iterations)
    
    # Initiates optimisation
    def optimise(self, iterations:int=5, block_size:int=40) -> None:
        self.__ctrl__.define_model(self.__model_name__)
        self.__ctrl__.define_param_mappers(self.__param_scale_dict__)
        self.__ctrl__.define_initial_values(self.__initial_param_dict__)
        self.__ctrl__.load_csv_files(self.__csv_file_list__)
        self.__ctrl__.define_data_mappers(self.__data_scale_dict__)
        self.__ctrl__.scale_and_process_data()
        self.__ctrl__.prepare(iterations, block_size)
        self.__ctrl__.display_initial_information()
        self.__ctrl__.optimise()
