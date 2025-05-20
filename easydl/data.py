import torch
from torch.utils.data import Dataset
import pandas as pd


class GenericPytorchDataset(Dataset):
    """
    A generic PyTorch Dataset class that takes a DataFrame and a dictionary of transformation functions.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        transforms (dict): A dictionary where keys are column names of the DataFrame,
            and values are functions that transform the raw values in those columns
            to PyTorch tensors.  If a column's key is not in the dict, the original value is passed as is.
    """
    def __init__(self, df, transforms=None):
        """
        Initializes the GenericDataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            transforms (dict, optional): Dictionary of transformations. Defaults to None.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self.df = df
        self.transforms = transforms if transforms is not None else {}
        self.columns = df.columns

    def __len__(self):
        """
        Returns the number of items in the dataset (i.e., the number of rows in the DataFrame).
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Fetches the data for a given index and applies the specified transformations.

        Args:
            index (int): Index of the data item to retrieve.

        Returns:
            dict: A dictionary where keys are column names and values are the
                corresponding transformed data (as PyTorch tensors, if a transform was applied).
        """
        row = self.df.iloc[index]
        data = {}
        for col in self.columns:
            value = row[col]
            if col in self.transforms:
                transform = self.transforms[col]
                try:
                    value = transform(value)  # Apply the transformation
                except Exception as e:
                    raise RuntimeError(f"Error applying transform to column '{col}' at index {index}: {e}") from e
            #check if it is already a tensor, if not convert it.
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            data[col] = value
        return data
    

