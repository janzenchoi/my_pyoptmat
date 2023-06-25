"""
 Title:         Converter
 Description:   For converting data into different formats
 Author:        Janzen Choi

 """

# Libraries
import xarray
import myoptmat.math.general as general

# Constants
DEFAULT_VALUES = {
    "time": 0,
    "strain": 0,
    "stress": 0,
    "temperature": 0,
    "cycle": 0,
    "type": "tensile",
    "control": "strain"
}

# Tries to float cast a value
def try_float_cast(value:str) -> float:
    try:
        return float(value)
    except:
        return value

# Converts a header file into a dict of lists
def csv_to_dict(csv_path:str, delimeter:str=",") -> dict:

    # Read all data from CSV (assume that file is not too big)
    csv_fh = open(csv_path, "r")
    csv_lines = csv_fh.readlines()
    csv_fh.close()

    # Initialisation for conversion
    csv_dict = {}
    headers = csv_lines[0].replace("\n", "").split(delimeter)
    csv_lines = csv_lines[1:]
    for header in headers:
        csv_dict[header] = []

    # Start conversion to dict
    for csv_line in csv_lines:
        csv_line_list = csv_line.replace("\n", "").split(delimeter)
        for i in range(len(headers)):
            value = csv_line_list[i]
            if value == "":
                continue
            value = try_float_cast(value)
            csv_dict[headers[i]].append(value)
    
    # Convert single item lists to items
    for header in headers:
        if len(csv_dict[header]) == 1:
            csv_dict[header] = csv_dict[header][0]
    
    # Return
    return csv_dict

# Converts a dictionary into a CSV file
def dict_to_csv(data_dict:dict, csv_path:str) -> None:
    
    # Extract headers and turn all values into lists
    headers = data_dict.keys()
    for header in headers:
        if not isinstance(data_dict[header], list):
            data_dict[header] = [data_dict[header]]
    
    # Open CSV file and write headers
    csv_fh = open(csv_path, "w+")
    csv_fh.write(",".join(headers) + "\n")
    
    # Write data and close
    max_list_size = max([len(data_dict[header]) for header in headers])
    for i in range(max_list_size):
        row_list = [str(data_dict[header][i]) if i < len(data_dict[header]) else "" for header in headers]
        row_str = ",".join(row_list)
        csv_fh.write(row_str + "\n")
    csv_fh.close()
    
# Checks the validity of a dictionary for conversion into a dataset
def check_dict(data_dict:dict) -> None:
    if not "time" in data_dict.keys() or not isinstance(data_dict["time"], list):
        raise ValueError(f"The data must contain a list of time values!")
    for header in data_dict.keys():
        if isinstance(data_dict[header], list) and len(data_dict[header]) != len(data_dict["time"]):
            raise ValueError(f"The lists must have the same number of data points!")

# Processes the data in a dictionary to conform to PyOptMat's format
def process_dict(data_dict:dict, num_points:int, default_values:dict=DEFAULT_VALUES) -> dict:
    
    # Initialise
    new_dict = {}
    thin_indexes = general.get_thin_indexes(len(data_dict["time"]), num_points)
    
    # Process the lists
    for header in ["time", "strain", "stress", "temperature", "cycle"]:
        if not header in data_dict.keys():
            new_dict[header] = [default_values[header]] * num_points
        elif isinstance(data_dict[header], list):
            new_dict[header] = [data_dict[header][i] for i in thin_indexes]
        else:
            new_dict[header] = [data_dict[header]] * num_points
            
    # Process non-list data
    for header in ["type", "control"]:
        if header in data_dict.keys():
            new_dict[header] = data_dict[header]
        else:
            new_dict[header] = default_values[header]
    return new_dict

# Converts a list of dictionaries into a dataset
def dict_list_to_dataset(dict_list:list, scale:float, num_points:int) -> xarray.core.dataset.Dataset:
    
    # Combine the dictionaries together
    combined_dict = {}
    for header in ["time", "strain", "stress", "temperature", "cycle"]:
        combined_dict[header] = []
        for i in range(num_points):
            row_data = [data_dict[header][i] for data_dict in dict_list]
            combined_dict[header].append(row_data)
    for header in ["type", "control"]:
        combined_dict[header] = [data_dict[header] for data_dict in dict_list]
        
    # Convert to dataset
    dataset = xarray.Dataset({
        "time":         (("ntime", "nexp"), combined_dict["time"]),
        "strain":       (("ntime", "nexp"), combined_dict["strain"]),
        "stress":       (("ntime", "nexp"), combined_dict["stress"]),
        "temperature":  (("ntime", "nexp"), combined_dict["temperature"]),
        "cycle":        (("ntime", "nexp"), combined_dict["cycle"]),
        "type":         (("nexp"), combined_dict["type"]),
        "control":      (("nexp"), combined_dict["control"])
    })
    
    # Add attributes and return
    dataset.attrs["scale"] = scale
    dataset.attrs["nrates"] = 1
    dataset.attrs["nsamples"] = len(dict_list)
    return dataset

# Converts an xarray into a dict (loses attributes)
def dataset_to_dict(dataset:xarray.core.dataset.Dataset, index:int=0) -> dict:
    list_dict = {}
    for key in ["time", "strain", "stress", "temperature", "cycle"]:
        list_dict[key] = list(dataset[key].transpose()[index].to_numpy())
    for key in ["type", "control"]:
        list_dict[key] = str(dataset[key][index].to_numpy())
    return list_dict
