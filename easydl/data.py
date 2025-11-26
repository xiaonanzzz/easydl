import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


class GenericLambdaDataset(Dataset):

    @staticmethod
    def list_to_lambda_loader(list) -> callable:
        return lambda index: list[index]

    """
    A generic PyTorch Dataset class that uses lambda functions to generate data.
    
    Args:
        lambda_dict (dict): A dictionary where keys are feature names and values are 
            lambda functions that take an index and return the corresponding value.
        length (int): The length of the dataset (required for __len__).
    """
    def __init__(self, lambda_dict, length):
        """
        Initializes the GenericLambdaDataset.
        
        Args:
            lambda_dict (dict): Dictionary of lambda functions. Each function should 
                accept an index (int) and return a value.
            length (int): The total number of items in the dataset.
        """
        if not isinstance(lambda_dict, dict):
            raise TypeError("lambda_dict must be a dictionary.")
        if not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative integer.")
        
        self.lambda_dict = lambda_dict
        self.length = length
    
    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return self.length
    
    def __getitem__(self, index):
        """
        Fetches the data for a given index by calling each lambda function with the index.
        
        Args:
            index (int): Index of the data item to retrieve.
            
        Returns:
            dict: A dictionary where keys are from lambda_dict and values are the 
                results of calling the corresponding lambda function with index.
        """
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} is out of range for dataset of length {self.length}")
        
        data = {}
        for key, lambda_func in self.lambda_dict.items():
            try:
                value = lambda_func(index)
                data[key] = value
            except Exception as e:
                raise RuntimeError(f"Error calling lambda function for key '{key}' at index {index}: {e}") from e
        
        return data


class GenericXYLambdaAutoLabelEncoderDataset(GenericLambdaDataset):
    def __init__(self, x_loader_lambda, y_loader_lambda, length):
        self.original_y_lambda = y_loader_lambda
        self.y_label_encoder = LabelEncoder()
        original_y_list = [y_loader_lambda(i) for i in range(length)]
        self.y_label_encoder.fit(original_y_list)
        transforms_dict = {
            'x': x_loader_lambda,
            'y': lambda y_index: self.y_label_encoder.transform([y_loader_lambda(y_index)])[0],
        }
        super().__init__(transforms_dict, length)

    def get_number_of_classes(self) -> int:
        return len(self.y_label_encoder.classes_)
    
    def get_y_list_with_encoded_labels(self) -> list:
        original_y_list = [self.original_y_lambda(i) for i in range(self.length)]
        encoded_y_list = self.y_label_encoder.transform(original_y_list)
        return list(encoded_y_list)