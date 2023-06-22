"""
 Title:         Controller
 Description:   Deals with calling the different components of the code
 Author:        Janzen Choi

"""

# Libraries
import torch
import myoptmat.interface.converter as converter
import myoptmat.interface.plotter as plotter
import myoptmat.interface.progressor as progressor
import myoptmat.interface.reader as reader
import myoptmat.math.mapper as mapper
import myoptmat.math.general as general
import myoptmat.models.__model__ as __model__

# Constants
HEADER_LIST = ["time", "strain", "stress", "temperature", "cycle"]

# Controller class
class Controller:
    
    # Constructor
    def __init__(self):
        
        # Initialise model variables
        self.model = None
        self.opt_model = None
        self.param_dict = {}
        
        # Initialise parameter variables
        self.param_mapper_dict = {}
        self.initial_param_list = []
        
        # Initialise data variables
        self.data_mapper_dict = {}
        self.csv_dict_list = []
        self.data_bounds_dict = {}
        self.dataset = None
        self.dataset_tuple = ()

    # Defines the model
    def define_model(self, model_name):
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

        # Pass mapper dictionary to model
        self.model.define_param_mapper_dict(self.param_mapper_dict)
    
    # Define dictionary of initial values
    def define_initial_values(self, initial_param_dict:dict) -> None:
        for param_name in self.param_dict.keys():
            if param_name in initial_param_dict.keys():
                initial_value = initial_param_dict[param_name]
            else:
                initial_value = self.param_mapper_dict[param_name].random()
            self.initial_param_list.append(initial_value)
    
    # Reads the CSV files as a list of dictionaries
    def load_csv_files(self, csv_file_list:list) -> None:
        
        # Load CSV data
        for csv_file in csv_file_list:
            self.csv_dict_list.append(converter.csv_to_dict(csv_file))
            
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
    def scale_data(self):
        for csv_dict in self.csv_dict_list:
            for header in HEADER_LIST:
                csv_dict[header] = self.data_mapper_dict[header].map(csv_dict[header])
        self.dataset = converter.dict_list_to_dataset(self.csv_dict_list)
        self.data, self.results, self.cycles, self.types, self.control = reader.load_dataset(self.dataset)
    
    # Prepare for the optimisation
    # TODO - allow user to choose different optimisers and objective functions
    def prepare(self, block_size:int, iterations:int) -> float:
        self.block_size = block_size
        self.iterations = iterations
        self.opt_model = self.model.get_opt_model(self.initial_param_list, self.block_size)
        self.algorithm = torch.optim.LBFGS(self.opt_model.parameters(), line_search_fn="strong_wolfe")
        self.mse_loss = torch.nn.MSELoss(reduction="sum")
    
    # Gets the predicted curves
    # TODO - should return x and y
    def get_predicted_curves(self):
        pass
    
    # Gets the optimal prediction
    # TODO - use torch.transpose(prediction, 0, 1)[0] to allow different input types
    def get_prediction(self):
        prediction = self.opt_model(self.data, self.cycles, self.types, self.control)
        prediction_mapper = self.data_mapper_dict["stress"]
        prediction = prediction_mapper.map(prediction)
        return prediction
    
    # Calculates the prediction discrepancy    
    def closure(self):
        self.algorithm.zero_grad()
        prediction = self.get_prediction()
        lossv = self.mse_loss(prediction, self.results)
        lossv.backward()
        return lossv
    
    # Conducts the optimisation
    def optimise(self, display:bool):
        
        # Optimise without displaying if desired
        if not display:
            for _ in range(self.iterations):
                self.algorithm.step(self.closure)
            return
        
        # Otherwise, display as we step
        general.print_value_list("Optimisation:", end="")
        pretext_format = "loss={}, "
        pv = progressor.ProgressVisualiser(self.iterations, pretext=pretext_format.format("?"))
        for _ in range(self.iterations):
            closure_loss = self.algorithm.step(self.closure)
            loss_value = "{:0.2}".format(closure_loss.detach().cpu().numpy())
            pv.progress(pretext=pretext_format.format(loss_value))
        pv.end()
    
    # Displays the parameter names
    def display_param_names(self):
        general.print_value_list("Parameters:", end="")
        print(f"[{', '.join([param_name for param_name in self.param_dict.keys()])}]")
        general.print_value_list("Initial Scaled:", self.initial_param_list)
    
    # Displays the initial gradient
    def display_initial_gradient(self):
        self.closure()
        gradients = [getattr(self.opt_model, param_name).grad for param_name in self.param_dict.keys()]
        gradients = [abs(g) for g in gradients]
        general.print_value_list("Initial Gradient:", gradients)

    # Displays the results of the optimisation
    def display_results(self):
        
        # Calculated unscaled parameters
        scaled_param_list = [float(getattr(self.opt_model, pn).data) for pn in self.param_dict.keys()]
        unscaled_param_list = []
        for i in range(len(scaled_param_list)):
            param_name = list(self.param_dict.keys())[i]
            param_mapper = self.param_mapper_dict[param_name]
            unscaled_param_list.append(param_mapper.unmap(scaled_param_list[i]))

        # Print parameters
        general.print_value_list("Final Scaled:", scaled_param_list)
        general.print_value_list("Final Unscaled:", unscaled_param_list)
        
        # Unscale the data
        csv_dict_list = [converter.dataset_to_dict(self.dataset, i) for i in range(self.dataset.nsamples)]
        for csv_dict in csv_dict_list:
            for header in ["time", "strain", "stress", "temperature", "cycle"]:
                data_mapper = self.data_mapper_dict[header]
                csv_dict[header] = [data_mapper.unmap(value) for value in csv_dict[header]]
        unscaled_dataset = converter.dict_list_to_dataset(csv_dict_list)
        
        # Plots the results
        prediction = self.get_prediction()
        prediction = self.data_mapper_dict["stress"].unmap(prediction)
        plt = plotter.Plotter("./plot", "strain", "stress")
        plt.plot_experimental(unscaled_dataset)
        plt.plot_prediction(unscaled_dataset, prediction)
        plt.save()
