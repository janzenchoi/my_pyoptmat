"""
 Title:         Optimiser
 Description:   Deals with the optimisation
 Author:        Janzen Choi

"""

# Libraries
import torch
import myoptmat.interface.converter as converter
import myoptmat.interface.plotter as plotter
import myoptmat.interface.progressor as progressor
import myoptmat.interface.reader as reader
import myoptmat.math.general as general
import myoptmat.math.mapper as mapper
import myoptmat.models.__model__ as __model__

# Optimiser class
class Optimiser:
    
    # Constructor
    def __init__(self, model_name:str, scale_params:bool=False, scale_data:bool=False):
        
        # Define model
        self.model = __model__.get_model(model_name)
        self.param_list = self.model.get_param_list()
        
        # Define scalers
        self.scale_params = scale_params
        self.scale_data = scale_data
        
        # Define internal variables
        self.initial_values = []
        self.dataset = None
        self.dataset_tuple = ()
        
        # Define mappers
        self.param_mapper_list = []
        self.data_mapper_dict = {}
    
    # Scale the parameters
    def initialise_param_mappers(self, param_scales:dict) -> None:
        pass
    
    # Initialises the parameters
    def initialise_params(self, initial_values:list=None) -> None:
        
        # Create parameter maps
        for param in self.param_list:
            l_bound = 0 if self.scale_params else param["l_bound"]
            u_bound = 1 if self.scale_params else param["u_bound"]
            param_mapper = mapper.Mapper(param["l_bound"], param["u_bound"], l_bound, u_bound)
            self.param_mapper_list.append(param_mapper)
        
        # Set initial values
        if initial_values == None:
            initial_values = [param_mapper.random() for param_mapper in self.param_mapper_list]
        elif len(self.param_list) != len(initial_values):
            raise TypeError("Number of initial conditions do not match the number of parameters!")
        self.initial_values = initial_values
    
    # Initialises the data
    def initialise_data(self, csv_file_list:list) -> None:
        
        # Read CSV files as a list of dictionaries
        csv_dict_list = [converter.csv_to_dict(csv_file) for csv_file in csv_file_list]
        
        # Create maps for the data
        self.data_mapper_dict = {}
        for header in ["time", "strain", "stress", "temperature", "cycle"]:
            
            # Combine and flatten data
            data_list = [csv_dict[header] for csv_dict in csv_dict_list]
            data_list = [item for sublist in data_list for item in sublist]
            
            # Define intervals
            in_l_bound = min(data_list)
            in_u_bound = max(data_list)
            out_l_bound = 0 if self.scale_data else in_l_bound
            out_u_bound = 1 if self.scale_data else in_u_bound
            
            # Create and append map
            data_mapper = mapper.Mapper(in_l_bound, in_u_bound, out_l_bound, out_u_bound)
            self.data_mapper_dict[header] = data_mapper            

        # Scale all the data
        for csv_dict in csv_dict_list:
            for header in ["time", "strain", "stress", "temperature", "cycle"]:
                data_mapper = self.data_mapper_dict[header]
                csv_dict[header] = [data_mapper.map(value) for value in csv_dict[header]]
                
        # Create dataset
        self.dataset = converter.dict_list_to_dataset(csv_dict_list)
        self.data, self.results, self.cycles, self.types, self.control = reader.load_dataset(self.dataset)
    
    # Initialise the settings
    def initialise_settings(self, block_size:int) -> None:
        self.opt_model = self.model.get_opt_model(self.initial_values, self.scale_params, block_size)
        self.algorithm = torch.optim.LBFGS(self.opt_model.parameters(), line_search_fn="strong_wolfe")
        self.mse_loss = torch.nn.MSELoss(reduction="sum")
    
    # # Gets the predicted curves
    # def get_predicted_curves(self):
    #     pass
    
    # Gets the optimal prediction
    def get_prediction(self):
        prediction = self.opt_model(self.data, self.cycles, self.types, self.control)
        prediction_mapper = self.data_mapper_dict["stress"]
        # print(torch.transpose(prediction, 0, 1)[0]) # need to map for different curves
        prediction = prediction_mapper.map(prediction)
        return prediction
    
    # Calculates the prediction discrepancy    
    def closure(self):
        self.algorithm.zero_grad()
        prediction = self.get_prediction()
        lossv = self.mse_loss(prediction, self.results)
        lossv.backward()
        return lossv
    
    # Gets the initial gradient
    # TODO - make me nicer
    def get_gradient(self):
        self.closure()
        print([float(getattr(self.opt_model, pn).grad) for pn in self.model.get_param_names()])
    
    # Conducts the optimisation
    def conduct_optimisation(self, iterations:int):
        pretext_format = "Optimising:     loss={}, "
        pv = progressor.ProgressVisualiser(iterations, pretext=pretext_format.format("?"))
        for _ in range(iterations):
            closure_loss = self.algorithm.step(self.closure)
            loss_value = "{:0.2}".format(closure_loss.detach().cpu().numpy())
            pv.progress(pretext=pretext_format.format(loss_value))
        pv.end()

    # Displays the results of the optimisation
    def display_results(self):
        
        # Calculated unscaled parameters
        scaled_params = [float(getattr(self.opt_model, pn).data) for pn in self.model.get_param_names()]
        unscaled_params = []
        for i in range(len(scaled_params)):
            param_mapper = self.param_mapper_list[i]
            unscaled_param = param_mapper.unmap(scaled_params[i])
            unscaled_params.append(unscaled_param)

        # Print results
        def print_list(value_list):
            str_list = ["{:0.3}".format(value) for value in value_list]
            print(f"[{', '.join(str_list)}]")
        print("Initial Scaled: ", end="")
        print_list(self.initial_values)
        print("Final Scaled:   ", end="")
        print_list(scaled_params)
        print("Final Unscaled: ", end="")
        print_list(unscaled_params)
        
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
