"""
 Title:         Optimiser
 Description:   Deals with the optimisation
 Author:        Janzen Choi

"""

# Libraries
import numpy as np, torch
import myoptmat.interface.converter as converter
import myoptmat.interface.plotter as plotter
import myoptmat.interface.progressor as progressor
import myoptmat.interface.reader as reader
import myoptmat.math.mapper as mapper
import myoptmat.models.__model__ as __model__

# Optimiser class
class Optimiser:
    
    # Constructor
    def __init__(self, model_name:str):
        self.model = __model__.get_model(model_name)
        self.param_list = self.model.get_param_list()
        self.initial_values = []
        self.dataset = None
        self.dataset_tuple = ()
        self.param_mapper_list = []
        self.data_mapper_dict = {}
    
    # Initialises the parameters
    def initialise_params(self, initial_values:list=None, scale_params:bool=False) -> None:
        
        # Create parameter maps
        for param in self.param_list:
            l_bound = 0 if scale_params else param["l_bound"]
            u_bound = 1 if scale_params else param["u_bound"]
            param_mapper = mapper.Mapper(param["l_bound"], param["u_bound"], l_bound, u_bound)
            self.param_mapper_list.append(param_mapper)
        
        # Set initial values
        if initial_values == None and scale_params:
            initial_values = [np.random.uniform(0, 1) for _ in self.param_list]
        elif initial_values == None and not scale_params:
            initial_values = [np.random.uniform(param["l_bound"], param["u_bound"]) for param in self.param_list]
        elif len(self.param_list) != len(initial_values):
            raise TypeError("Number of initial conditions do not match the number of parameters!")
        self.initial_values = initial_values
        
    # Initialises the data
    def initialise_data(self, csv_file_list:list, scale_data:bool=False) -> None:
        
        # Read CSV files as a list of dictionaries
        csv_dict_list = [converter.csv_to_dict(csv_file) for csv_file in csv_file_list]
        
        # # Create maps for the data
        # self.data_mapper_dict = {}
        # for header in ["time", "strain", "stress", "temperature", "cycle"]:
            
        #     # Combine and flatten data
        #     data_list = [csv_dict[header] for csv_dict in csv_dict_list]
        #     data_list = [item for sublist in data_list for item in sublist]
            
        #     # Define intervals
        #     in_l_bound = min(data_list)
        #     in_u_bound = max(data_list)
        #     out_l_bound = 0 if scale_data else in_l_bound
        #     out_u_bound = 1 if scale_data else in_u_bound
            
        #     # Create and append map
        #     data_mapper = mapper.Mapper(in_l_bound, in_u_bound, out_l_bound, out_u_bound)
        #     self.data_mapper_dict[header] = data_mapper            

        # # Scale all the data
        # for csv_dict in csv_dict_list:
        #     for header in ["time", "strain", "stress", "temperature", "cycle"]:
        #         data_mapper = self.data_mapper_dict[header]
        #         csv_dict[header] = [data_mapper.map(value) for value in csv_dict[header]]
                
        # Create dataset
        self.dataset = converter.dict_list_to_dataset(csv_dict_list)
        self.data, self.results, self.cycles, self.types, self.control = reader.load_dataset(self.dataset)
    
    # Initialise the settings
    def initialise_settings(self, block_size:int) -> None:
        self.opt_model = self.model.get_opt_model(self.initial_values, block_size)
        self.algorithm = torch.optim.LBFGS(self.opt_model.parameters())
        self.mse_loss = torch.nn.MSELoss(reduction="sum")
    
    # Gets the optimal prediction
    def get_prediction(self):
        return self.opt_model(self.data, self.cycles, self.types, self.control)
    
    # Calculates the prediction discrepancy    
    def closure(self):
        self.algorithm.zero_grad()
        lossv = self.mse_loss(self.get_prediction(), self.results)
        lossv.backward()
        return lossv
    
    # Conducts the optimisation
    def conduct_optimisation(self, iterations:int):
        
        # Optimise and visualise
        pretext_format = "Optimising:     loss={}, "
        pv = progressor.ProgressVisualiser(iterations, pretext=pretext_format.format("?"))
        for _ in range(iterations):
            closure_loss = self.algorithm.step(self.closure)
            loss_value = "{:0.2}".format(closure_loss.detach().cpu().numpy())
            pv.progress(pretext=pretext_format.format(loss_value))
        pv.end()

        # Calculated unscaled parameters
        scaled_params = [float(getattr(self.opt_model, pn).data) for pn in self.model.get_param_names()]
        unscaled_params = []
        for i in range(len(scaled_params)):
            param_mapper = self.param_mapper_list[i]
            unscaled_param = param_mapper.unmap(scaled_params[i])
            unscaled_params.append(unscaled_param)

        # Print results
        print(f"Initial Scaled: {self.initial_values}")
        print(f"Final Scaled:   {scaled_params}")
        print(f"Final Unscaled: {unscaled_params}")
        
        # Plots the results
        prediction = self.get_prediction()
        plt = plotter.Plotter("./plot", "strain", "stress")
        plt.plot_experimental(self.dataset)
        plt.plot_prediction(self.dataset, prediction)
        plt.save()
