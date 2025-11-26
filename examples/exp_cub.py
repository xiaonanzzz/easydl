from datasets import load_dataset
from easydl.data import GenericLambdaDataset, GenericXYLambdaAutoLabelEncoderDataset
from easydl.dml.trainer import DeepMetricLearningImageTrainverV871, DeepMetricLearningImageTrainverV971
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.evaluation import evaluate_embedding_top1_accuracy_ignore_self
from easydl.dml.infer import images_to_embeddings_one_by_one
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from easydl.image import CommonImageToDlTensorForTraining
import os
from pathlib import Path
from easydl.common_infer import infer_x_dataset_with_simple_stacking

class ExpCubConfig:
    embedding_dim = 128
    batch_size = 256
    num_epochs = 2
    lr = 1e-4

def train_main():
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds['train'][i]['image']
    y_loader = lambda i: ds['train'][i]['text']
    DeepMetricLearningImageTrainverV871.train_resnet18_with_arcface_loss(x_loader, y_loader, len(ds['train']), embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)


class WorkingDirManager:
    def __init__(self, exp_dir_path):
        self.exp_dir_path = Path(exp_dir_path)
        self.original_working_dir_path = os.getcwd()

    def swtich_to_exp_dir(self):
        self.exp_dir_path.mkdir(parents=True, exist_ok=True)
        os.chdir(self.exp_dir_path)

    def switch_back_to_original_working_dir(self):
        os.chdir(self.original_working_dir_path)


def exp_cub_v971():

    # prepare output directory
    working_dir_manager = WorkingDirManager('exp_cub_v971_tmp')
    working_dir_manager.swtich_to_exp_dir()

    # prepare dataset
    ds = load_dataset("cassiekang/cub200_dataset")
    image_item_to_tensor_transform = CommonImageToDlTensorForTraining()
    x_loader = lambda i: image_item_to_tensor_transform(ds['train'][i]['image'])
    y_loader = lambda i: ds['train'][i]['text']
    ds_train = GenericXYLambdaAutoLabelEncoderDataset(x_loader, y_loader, len(ds['train']))
    # train model
    DeepMetricLearningImageTrainverV971(ds_train, ds_train.get_number_of_classes(), model_name='resnet18', loss_name='arcface_loss', embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)



    # evaluate model
    for epoch in range(1, ExpCubConfig.num_epochs + 1):
        model_path = f'model_epoch_{epoch:03d}.pth'
        if os.path.exists(model_path):
            results = evaluate_one_epoch(model_path)
            print(f"Epoch {epoch}: {results}")
            os.remove(model_path)
    
    working_dir_manager.switch_back_to_original_working_dir()

def evaluate_one_epoch(model_path):
    """
    Evaluate a trained ResNet18 model on the test set of CUB dataset.
    
    Args:
        model_path: Path to the saved model checkpoint (.pth file)
    
    Returns:
        dict: Dictionary containing evaluation results with keys:
            - avg_top1_accuracy: Average top1 accuracy
            - accuracy_upper_bound: Upper bound of top1 accuracy
            - result_dataframe: DataFrame with detailed results
    """
    # Load dataset
    ds = load_dataset("cassiekang/cub200_dataset")
    test_size = len(ds['test'])
    
    # Encode labels for test set
    y_list = [ds['test'][i]['text'] for i in range(test_size)]
    y_encoder = LabelEncoder()
    encoded_y_list = y_encoder.fit_transform(y_list)
    
    # Get list of PIL images from test set
    test_images = [ds['test'][i]['image'] for i in range(test_size)]
    
    # Load model as ImageModelWrapper
    model_wrapper = Resnet18MetricModel.create_image2vector_wrapper(embedding_dim=ExpCubConfig.embedding_dim, model_param_path=model_path)
    
    # Get embeddings using images_to_embeddings_one_by_one
    print("Running inference on test set...")
    all_embeddings = images_to_embeddings_one_by_one(test_images, model_wrapper)
    
    # Create dataframe for evaluation
    embeddings_df = pd.DataFrame({
        'embedding': [emb for emb in all_embeddings],
        'label': encoded_y_list
    })
    
    # Evaluate
    print("Computing evaluation metrics...")
    results = evaluate_embedding_top1_accuracy_ignore_self(embeddings_df)
    
    print(f"Top-1 Accuracy: {results['avg_top1_accuracy']:.4f}")
    print(f"Accuracy Upper Bound: {results['accuracy_upper_bound']:.4f}")
    
    return results


def get_test_dataset_with_encoded_labels() -> GenericXYLambdaAutoLabelEncoderDataset:
    """
    Get the test dataset and encoded labels.
    """
    # prepare dataset
    ds = load_dataset("cassiekang/cub200_dataset")
    image_item_to_tensor_transform = CommonImageToDlTensorForTraining()
    x_loader = lambda i: image_item_to_tensor_transform(ds['test'][i]['image'])
    y_loader = lambda i: ds['test'][i]['text']
    ds_test = GenericXYLambdaAutoLabelEncoderDataset(x_loader, y_loader, len(ds['test']))
    return ds_test


def exp_eval_pretrained_resenet18():
    """
    Evaluate a pretrained ResNet18 model on the test set of CUB dataset.
    """
    # Load dataset
    ds_test = get_test_dataset_with_encoded_labels()
    
    # Load model as ImageModelWrapper
    model_wrapper = Resnet18MetricModel(128)
    
    # Get embeddings using images_to_embeddings_one_by_one
    print("Running inference on test set...")
    all_embeddings = infer_x_dataset_with_simple_stacking(ds_test, model_wrapper)
    
    # Create dataframe for evaluation
    embeddings_df = pd.DataFrame({
        'embedding': [emb for emb in all_embeddings],
        'label': ds_test.get_y_list_with_encoded_labels()
    })
    
    # Evaluate
    print("Computing evaluation metrics...")
    results = evaluate_embedding_top1_accuracy_ignore_self(embeddings_df)
    
    print(f"Top-1 Accuracy: {results['avg_top1_accuracy']:.4f}")
    print(f"Accuracy Upper Bound: {results['accuracy_upper_bound']:.4f}")
    
    return results


if __name__ == "__main__":
    # evaluate_one_epoch('model_epoch_001.pth')

    exp_eval_pretrained_resenet18()