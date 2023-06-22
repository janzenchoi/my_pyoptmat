"""
 Title:         Plotter
 Description:   For visualising results
 Author:        Janzen Choi

 """

# Libraries
import xarray, torch
import matplotlib.pyplot as plt
import myoptmat.interface.converter as converter
import myoptmat.math.units as units

# Plotter class
class Plotter:

    # Constructor
    def __init__(self, path:str, x_label:str, y_label:str, x_units:str=None, y_units:str=None):
        
        # Initialise internal variables
        self.path = path
        self.x_label = x_label
        self.y_label = y_label
        self.x_units = units.MYOPTMAT_UNITS[self.x_label] if x_units == None else x_units
        self.y_units = units.MYOPTMAT_UNITS[self.y_label] if y_units == None else y_units
        
        # Prepare plot
        plt.xlabel(f"{self.x_label.capitalize()} ({self.x_units})")
        plt.ylabel(f"{self.y_label.capitalize()} ({self.y_units})")
        plt.title("Experimental vs Prediction")
        plt.plot([], [], color="red") # prediction
        plt.scatter([], [], color="silver") # experimental
        plt.legend(["Prediction", "Experimental"])

    # Plots the experimental data
    def plot_experimental(self, dataset:xarray.core.dataset.Dataset) -> None:
        num_data = dataset.nrates * dataset.nsamples
        for i in range(num_data):
            data_dict = converter.dataset_to_dict(dataset, i)
            x_list = data_dict[self.x_label]
            y_list = data_dict[self.y_label]
            plt.scatter(x_list, y_list, color="silver")
    
    # Plots the predicted data
    def plot_prediction(self, dataset:xarray.core.dataset.Dataset, results:torch.Tensor) -> None:
        num_data = dataset.nrates * dataset.nsamples
        for i in range(num_data):
            data_dict = converter.dataset_to_dict(dataset, i)
            x_list = data_dict[self.x_label]
            y_list = [t[i] for t in results.tolist()]
            plt.plot(x_list, y_list, color="red")
    
    # Saves and clears the plot
    def save(self) -> None:
        plt.savefig(self.path)
        plt.cla()
