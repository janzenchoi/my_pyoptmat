"""
 Title:         Recorder
 Description:   Records results
 Author:        Janzen Choi

"""

# Libraries
import pandas as pd

# Recorder class
class Recorder:
    
    # Constructor
    def __init__(self, results_path:str):
        self.results_path = results_path
    
    # Creates a new file
    def create_new_file(self, curr_iteration:int):
        results_path = f"{self.results_path}_{curr_iteration}.xlsx"
        self.writer = pd.ExcelWriter(results_path, engine="xlsxwriter")
        
    # Writes a dictionary of lists to a sheet with nice formatting
    def write_data(self, data_dict:dict, sheet_name:str) -> None:
        
        # Convert dictionary to dataframe
        columns = list(data_dict.keys())
        data = zip_longest([data_dict[column] for column in columns])
        data = list(map(list, zip(*data)))
        dataframe = pd.DataFrame(data, columns = columns)
        
        # Apply fit column widths
        sheet_name = "results"
        dataframe.style.apply(centre_align, axis=0).to_excel(self.writer, sheet_name, index = False)
        sheet = self.writer.sheets[sheet_name]
        for column in dataframe:
            column_length = max(dataframe[column].astype(str).map(len).max(), len(column)) + 1
            column_index = dataframe.columns.get_loc(column)
            sheet.set_column(column_index, column_index, column_length)
    
    # Writes data and the associated plot given a dictionary of dictionaries
    #   Dictionary requires 'x', 'y', 'size', and 'colour' keys
    def write_plot(self, data_dict_dict:dict, sheet_name:str, x_label:str, y_label:str, plot_type:str) -> None:
        
        # Convert dictionary of dictionary into list of lists
        data_list_list = []
        for data_name in data_dict_dict.keys():
            data_dict = data_dict_dict[data_name]
            if len(data_dict["x"]) == 0 or len(data_dict["y"]) == 0:
                continue
            data_list_list += [data_dict["x"], data_dict["y"]]
        
        # Convert data into pandas' desired format
        data_list_map = list(map(list, zip(*data_list_list)))
        pd.DataFrame(data_list_map).to_excel(self.writer, sheet_name, index=False)
        
        # Add sheet to spreadsheet and create chart
        sheet = self.writer.sheets[sheet_name]
        chart = self.writer.book.add_chart({"type": plot_type})
        
        # Add converted data to a chart
        for i in range(len(data_dict_dict.keys())):
            data_dict = data_dict_dict[list(data_dict_dict.keys())[i]]
            marker_style = {
                "type": "circle",
                "size": data_dict["size"]
            }
            chart.add_series({
                "name":       list(data_dict_dict.keys())[i],
                "categories": [sheet_name, 1, i*2, len(data_dict["x"]), i*2],
                "values":     [sheet_name, 1, i*2+1, len(data_dict["x"]), i*2+1],
                "marker":     marker_style
            })
        
        # Add axes and add the chart to the sheet
        chart.set_x_axis({"name": x_label, "major_gridlines": {"visible": True}})
        chart.set_y_axis({"name": y_label, "major_gridlines": {"visible": True}})
        sheet.insert_chart("A1", chart)

    # Closes the writer
    def close(self):
        self.writer.close()

# For centre-aligning the cellss
def centre_align(x:float):
    return ["text-align: center" for _ in x]

# Imitates zip longest but for a list of lists
def zip_longest(list_list:list) -> list:
    max_values = max([len(list) for list in list_list])
    new_list_list = []
    for list in list_list:
        new_list = list + [None] * (max_values - len(list))
        new_list_list.append(new_list)
    return new_list_list