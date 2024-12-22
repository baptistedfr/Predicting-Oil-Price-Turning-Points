import numpy as np
import pandas as pd

def load_data(file_path: str) -> np.ndarray:
    """
    Load historical price data from a CSV file and return a NumPy array.

    The CSV is assumed to have date information in row 4 onward,
    which is skipped by 'skiprows=4'. The resulting dataset will have
    two columns: [Date, Price]. The 'Date' column is converted to
    np.datetime64[D], then the data is sorted by date.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 2), where:
        - Column 0 is the date as np.datetime64[D].
        - Column 1 is the price as a float.
    """
    # Load data using Pandas
    data = pd.read_csv(file_path, skiprows=4)
    data.columns = ["Date", "Price"]

    # Convert 'Date' to datetime64[D]
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

    # Sort by date
    data = data.sort_values(by="Date")

    # Convert to NumPy array
    return data.to_numpy()


def select_sample(data: np.ndarray, time_start: str, time_end: str) -> np.ndarray:
    """
    Select a date range from the loaded data and convert the dates to a numeric index.

    The function assumes the data has shape (N, 2), with:
    - Column 0 = Date (np.datetime64[D])
    - Column 1 = Price (float).

    The time_start/time_end arguments are strings in '%m/%d/%Y' format, which
    will be converted to np.datetime64. Then, the selected subset of data is
    retrieved by date filtering. The date column is replaced by a linearly
    spaced numeric array [1..length of subset].

    Parameters
    ----------
    data : np.ndarray
        A 2D array of shape (N, 2). First column is Date as np.datetime64[D].
    time_start : str
        The start date in '%m/%d/%Y' format (e.g. "01/01/2020").
    time_end : str
        The end date in '%m/%d/%Y' format (e.g. "12/31/2020").

    Returns
    -------
    np.ndarray, shape (M, 2)
        The filtered dataset, where:
        - Column 0 is an integer index from 1..M.
        - Column 1 is the Price (float).
        Converted to float dtype.
    """
    # Convert time_start and time_end to datetime64
    start_dt = np.datetime64(pd.to_datetime(time_start, format="%m/%d/%Y"))
    end_dt   = np.datetime64(pd.to_datetime(time_end, format="%m/%d/%Y"))
    
    # Create a boolean mask for the date range
    mask = (data[:, 0] >= start_dt) & (data[:, 0] <= end_dt)
    sample_data = data[mask]

    # Replace the date column with a linearly spaced index
    sample_data_count = len(sample_data)
    sample_data[:, 0] = np.linspace(1, sample_data_count, sample_data_count)

    return sample_data.astype(float)


def convert_param_bounds(param_bounds_dict: dict) -> np.ndarray:
    """
    Convert a dictionary of parameter bounds into a 2D NumPy array.

    The dictionary is expected to have keys: 't_c', 'omega', 'phi', 'alpha',
    each mapped to a tuple or list [low, high].

    The resulting array has shape (4, 2), where each row corresponds to
    [low, high] for one parameter in the order:
        1. t_c
        2. omega
        3. phi
        4. alpha

    Parameters
    ----------
    param_bounds_dict : dict
        A dictionary containing the bounds for each parameter, e.g.:
            {
                "t_c": (0, 3650),
                "omega": (0, 40),
                "phi": (0, 2*np.pi),
                "alpha": (0.1, 0.9)
            }

    Returns
    -------
    np.ndarray, shape (4, 2)
        A float64 array of bounds in the order [t_c, omega, phi, alpha].
        Each row is [min, max].
    """
    param_bounds_array = np.array([
        [param_bounds_dict["t_c"][0],    param_bounds_dict["t_c"][1]],
        [param_bounds_dict["omega"][0],  param_bounds_dict["omega"][1]],
        [param_bounds_dict["phi"][0],    param_bounds_dict["phi"][1]],
        [param_bounds_dict["alpha"][0],  param_bounds_dict["alpha"][1]]
    ], dtype=np.float64)
    
    return param_bounds_array