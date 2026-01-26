from datasets import load_dataset

from easydl.data import GenericXYLambdaAutoLabelEncoderDataset


def get_train_dataset_with_image_and_encoded_labels() -> (
    GenericXYLambdaAutoLabelEncoderDataset
):
    """
    Get the train dataset and encoded labels.
    """
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds["train"][i]["image"]
    y_loader = lambda i: ds["train"][i]["text"]
    ds_train = GenericXYLambdaAutoLabelEncoderDataset(
        x_loader, y_loader, len(ds["train"])
    )
    return ds_train


def get_test_dataset_with_image_and_encoded_labels() -> (
    GenericXYLambdaAutoLabelEncoderDataset
):
    """
    Get the test dataset and encoded labels.
    """
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds["test"][i]["image"]
    y_loader = lambda i: ds["test"][i]["text"]
    ds_test = GenericXYLambdaAutoLabelEncoderDataset(
        x_loader, y_loader, len(ds["test"])
    )
    return ds_test


def get_small_train_dataset_with_image_and_encoded_labels(
    num_samples: int = 200,
) -> GenericXYLambdaAutoLabelEncoderDataset:
    """
    Get the small train dataset and encoded labels.
    """
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds["train"][i]["image"]
    y_loader = lambda i: ds["train"][i]["text"]
    ds_train = GenericXYLambdaAutoLabelEncoderDataset(x_loader, y_loader, num_samples)
    return ds_train
