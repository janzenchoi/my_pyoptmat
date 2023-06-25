"""
 Title:         Controller
 Description:   Deals with calling the different components of the code
 Author:        Janzen Choi

"""

# Libraries
import torch, copy
import myoptmat.interface.converter as converter
import myoptmat.interface.progressor as progressor
import myoptmat.interface.reader as reader
import myoptmat.interface.recorder as recorder
import myoptmat.math.mapper as mapper
import myoptmat.math.general as general
import myoptmat.models.__model__ as __model__

# Constants
NUM_POINTS = 50
HEADER_LIST = ["time", "strain", "stress", "temperature", "cycle"]

# Controller class
class Controller:
    
    # Constructor
    def __init__(self):
        
        # Initialise model variables
        self.model = None
        self.block_size = None
        self.iterations = None
        self.opt_model = None
        self.algorithm = None
        self.loss_function = None
        self.param_dict = {}
        
        # Initialise parameter variables
        self.param_mapper_dict = {}
        self.initial_param_list = []
        self.param_bound_list = []
        self.param_scale_list = []
        
        # Initialise data variables
        self.data_mapper_dict = {}
        self.csv_dict_list = []
        self.raw_csv_dict_list = []
        self.data_bounds_dict = {}
        
        # Initialise transformed data variables
        self.dataset = None
        self.data = None
        self.results = None
        self.cycles = None
        self.types = None
        self.control = None

        # Initialise result variables
        self.loss_value_list = []
        self.recorder = None
        self.record_iterations = None

    # Defines the model
    def define_model(self, model_name:str):
        self.model = __model__.get_model(model_name)
        self.param_dict = self.model.get_param_dict()
    
    # Define parameter mappers
    def define_param_mappers(self, param_scale_dict:dict) -> None:
        
        # Iterate through parameters
        for param_name in self.param_dict.keys():
            
            # Define bounds based on whether the user has defined bounds or not
            is_defined = param_name in param_scale_dict.keys()
            in_l_bound = self.param_dict[param_name]["l_bound"]
            in_u_bound = self.param_dict[param_name]["u_bound"]
            out_l_bound = param_scale_dict[param_name]["l_bound"] if is_defined else in_l_bound
            out_u_bound = param_scale_dict[param_name]["u_bound"] if is_defined else in_u_bound
            
            # Create parameter mapper and add
            param_mapper = mapper.Mapper(in_l_bound, in_u_bound, out_l_bound, out_u_bound)
            self.param_mapper_dict[param_name] = param_mapper

        # Get bound / scale summary
        self.param_bound_list, self.param_scale_list = [], []
        for param_name in self.param_dict.keys():
            param_mapper = self.param_mapper_dict[param_name]
            in_bounds = param_mapper.get_in_bounds()
            self.param_bound_list.append(f"[{in_bounds[0]}, {in_bounds[1]}]")
            out_bounds = param_mapper.get_out_bounds()
            self.param_scale_list.append(f"[{out_bounds[0]}, {out_bounds[1]}]")

        # Pass mapper dictionary to model
        self.model.define_param_mapper_dict(self.param_mapper_dict)
    
    # Define dictionary of initial values
    def define_initial_values(self, initial_param_dict:dict) -> None:
        for param_name in self.param_dict.keys():
            if param_name in initial_param_dict.keys():
                initial_value = initial_param_dict[param_name]
                initial_value = self.param_mapper_dict[param_name].map(initial_value)
            else:
                initial_value = self.param_mapper_dict[param_name].random()
            self.initial_param_list.append(initial_value)
    
    # Reads the CSV files as a list of dictionaries
    def load_csv_files(self, csv_file_list:list) -> None:
        
        # Load CSV data
        for csv_file in csv_file_list:
            csv_dict = converter.csv_to_dict(csv_file)
            self.csv_dict_list.append(csv_dict)
            
        # Define bounds for each header
        for header in HEADER_LIST:
            data_list = [csv_dict[header] for csv_dict in self.csv_dict_list]
            data_list = [item for sublist in data_list for item in sublist]
            self.data_bounds_dict[header] = {"l_bound": min(data_list), "u_bound": max(data_list)}
    
    # Define data mappers
    def define_data_mappers(self, data_scale_dict:dict=None) -> None:

        # Iterate through data headers
        for header in HEADER_LIST:
            
            # Define bounds based on whether the user has defined bounds or not
            is_defined = header in data_scale_dict.keys()
            in_l_bound = self.data_bounds_dict[header]["l_bound"]
            in_u_bound = self.data_bounds_dict[header]["u_bound"]
            out_l_bound = data_scale_dict[header]["l_bound"] if is_defined else in_l_bound
            out_u_bound = data_scale_dict[header]["u_bound"] if is_defined else in_u_bound

            # Create data mapper and add
            data_mapper = mapper.Mapper(in_l_bound, in_u_bound, out_l_bound, out_u_bound)
            self.data_mapper_dict[header] = data_mapper
    
    # Scale the data
    def scale_and_process_data(self):
                
        # Checks and process dictionaries
        [converter.check_dict(data_dict) for data_dict in self.csv_dict_list]
        self.csv_dict_list = [converter.process_dict(data_dict, NUM_POINTS) for data_dict in self.csv_dict_list]
        self.raw_csv_dict_list = copy.deepcopy(self.csv_dict_list) # make a copy
        
        # Scale the data
        for csv_dict in self.csv_dict_list:
            for header in HEADER_LIST:
                csv_dict[header] = self.data_mapper_dict[header].map(csv_dict[header])
    
        # Convert to datasets
        self.dataset = converter.dict_list_to_dataset(self.csv_dict_list, 1, NUM_POINTS)
        self.data, self.results, self.cycles, self.types, self.control = reader.load_dataset(self.dataset)
    
    # Prepare for the optimisation
    # TODO - allow user to choose different optimisers and objective functions
    def prepare(self, iterations:int, block_size:int) -> float:
        
        # Define optimisation parameters
        self.iterations = iterations
        self.block_size = block_size
        
        # Get deterministic model
        self.opt_model = self.model.get_opt_model(self.initial_param_list, self.block_size)
        params = self.opt_model.parameters()
        
        # Define algorithm and loss functions
        # self.algorithm = torch.optim.LBFGS(params)
        self.algorithm = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")
        # self.algorithm = torch.optim.Adam(params)
        self.loss_function = torch.nn.MSELoss(reduction="sum")
    
    # Gets the optimal prediction
    # TODO - use torch.transpose(prediction, 0, 1)[0] to allow different input types
    def get_prediction(self) -> torch.tensor:
        prediction = self.opt_model(self.data, self.cycles, self.types, self.control)
        prediction_mapper = self.data_mapper_dict["stress"]
        prediction = prediction_mapper.map(prediction)
        return prediction
    
    # Calculates the prediction discrepancy    
    def closure(self) -> float:
        self.algorithm.zero_grad()
        prediction = self.get_prediction()
        lossv = self.loss_function(prediction, self.results)
        lossv.backward()
        return lossv

    # Initialises the recorder
    def initialise_recorder(self, record_path:str, record_iterations:int) -> None:
        self.recorder = recorder.Recorder(record_path)
        self.record_iterations = record_iterations
    
    # Get current unscaled optimal parameters
    def get_opt_params(self) -> list:
        opt_param_list = []
        for param_name in self.param_dict.keys():
            scaled_param = float(getattr(self.opt_model, param_name).data)
            param_mapper = self.param_mapper_dict[param_name]
            unscaled_param = param_mapper.unmap(scaled_param)
            opt_param_list.append(unscaled_param)
        return opt_param_list
    
    # Get experimental and predicted data (based on optimal parameters)
    def get_exp_prd_data(self, x_label:str, y_label:str) -> tuple:
        
        # Get prediction
        prediction = self.get_prediction()
        prediction = self.data_mapper_dict[y_label].unmap(prediction)

        # Get experimental and predicted data
        exp_x_list, exp_y_list, prd_y_list = [], [], []
        for i in range(len(self.raw_csv_dict_list)):
            exp_x_list += self.raw_csv_dict_list[i][x_label]
            exp_y_list += self.raw_csv_dict_list[i][y_label]
            prd_y_list += [p[i] for p in prediction.tolist()]
        
        # Return data
        return exp_x_list, exp_y_list, prd_y_list

    # Conducts the optimisation
    def optimise(self) -> None:
        
        # Initialise optimisation
        general.print_value_list("Optimisation:", end="")
        pv = progressor.ProgressVisualiser(self.iterations, pretext="loss=?, ")
        for curr_iteration in range(1, self.iterations+1):
            
            # Take a step, add loss to history, and print loss
            closure_loss = self.algorithm.step(self.closure)
            loss_value = "{:0.2}".format(closure_loss.detach().cpu().numpy())
            pv.progress(pretext=f"loss={loss_value}, ")
            self.loss_value_list.append(float(loss_value))
            
            # If recorder initialised and iterations reached, then record results
            if self.recorder != None and curr_iteration % self.record_iterations == 0:
                self.record_results(curr_iteration)
        
        # End optimisation
        opt_params = self.get_opt_params()
        general.print_value_list("Final Params:", opt_params)
        pv.end()

    # Runs each step of the optimisation
    def record_results(self, curr_iteration:int) -> None:
        
        # Initialise
        curr_iteration = str(curr_iteration).zfill(len(str(self.iterations)))
        self.recorder.create_new_file(curr_iteration)
        x_label, y_label = "strain", "stress"

        # Write the parameter results
        self.recorder.write_data({
            "parameter":    list(self.param_dict.keys()),
            "bounds":       self.param_bound_list,
            "scales":       self.param_scale_list,
            "optimised":    self.get_opt_params(),
        }, "results")
        
        # Plot experimental and predicted data
        exp_x_list, exp_y_list, prd_y_list = self.get_exp_prd_data(x_label, y_label)
        self.recorder.write_plot({
            "experimental": {"x": exp_x_list, "y": exp_y_list, "size": 5},
            "predicted":    {"x": exp_x_list, "y": prd_y_list, "size": 3}
        }, "plot", x_label, y_label, "scatter")
        
        # Plots the loss history
        loss_x_list = list(range(1, len(self.loss_value_list)+1))
        self.recorder.write_plot({
            "loss history": {"x": loss_x_list, "y": self.loss_value_list, "size": 3}
        }, "loss", "iteration", "loss", "line")
        
        # Saves the file
        self.recorder.close()
    
    # Displays the initial parameters and initial gradient
    def display_initial_information(self):
        
        # Display initial parameters
        initial_unscaled_param_list = []
        for i in range(len(self.param_dict.keys())):
            param_name = list(self.param_dict.keys())[i]
            param_mapper = self.param_mapper_dict[param_name]
            unscaled_param = param_mapper.unmap(self.initial_param_list[i])
            initial_unscaled_param_list.append(unscaled_param)
        general.print_value_list("Initial Params:", initial_unscaled_param_list)
        
        # Print initial gradient
        self.closure()
        gradients = [getattr(self.opt_model, param_name).grad for param_name in self.param_dict.keys()]
        gradients = [abs(g) for g in gradients]
        general.print_value_list("Initial Gradient:", gradients)
