from datasets import load_dataset
from easydl.data import GenericLambdaDataset
from easydl.dml.trainer import DeepMetricLearningImageTrainverV871
from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.evaluation import evaluate_embedding_top1_accuracy_ignore_self
from easydl.dml.infer import images_to_embeddings_one_by_one
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class ExpCubConfig:
    embedding_dim = 128
    batch_size = 256
    num_epochs = 4
    lr = 1e-4

def train_main():
    ds = load_dataset("cassiekang/cub200_dataset")
    x_loader = lambda i: ds['train'][i]['image']
    y_loader = lambda i: ds['train'][i]['text']
    DeepMetricLearningImageTrainverV871.train_resnet18_with_arcface_loss(x_loader, y_loader, len(ds['train']), embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)


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




if __name__ == "__main__":
    evaluate_one_epoch('model_epoch_001.pth')