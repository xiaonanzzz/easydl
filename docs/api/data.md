# easydl.data

Dataset classes for flexible data loading in PyTorch.

## Classes

::: easydl.data.GenericLambdaDataset
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__
        - from_list
        - from_xy_lists
        - from_dataframe
        - get_value_from_key_and_index
        - extend_lambda_dict
        - list_to_lambda_loader

::: easydl.data.GenericPytorchDataset
    options:
      show_root_heading: true
      show_source: true

::: easydl.data.GenericXYLambdaAutoLabelEncoderDataset
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_df
        - get_number_of_classes
        - get_y_list_with_encoded_labels
        - get_original_y_list
        - get_original_y_from_index
