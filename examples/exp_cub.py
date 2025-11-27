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
from easydl.image import CommonImageToDlTensorForTraining, COMMON_IMAGE_PREPROCESSING_FOR_TESTING
import os
from pathlib import Path
from easydl.common_infer import infer_x_dataset_with_simple_stacking
from easydl.utils import AcceleratorSetting
from sklearn.metrics.pairwise import cosine_similarity
from easydl.dml.evaluation import create_pairwise_similarity_ground_truth_matrix, evaluate_pairwise_score_matrix_with_true_label, calculate_cosine_similarity_matrix
from easydl.utils import WorkingDirManager, MainCmdManager
from easydl.dml.evaluation import DeepMetricLearningImageEvaluatorOnEachEpoch

class ExpCubConfig:
    embedding_dim = 384
    batch_size = 256
    num_epochs = 100
    lr = 1e-4


def get_test_dataset_with_encoded_labels() -> GenericXYLambdaAutoLabelEncoderDataset:
    """
    Get the test dataset and encoded labels.
    """
    # prepare dataset
    ds = load_dataset("cassiekang/cub200_dataset")
    image_item_to_tensor_transform = COMMON_IMAGE_PREPROCESSING_FOR_TESTING
    x_loader = lambda i: image_item_to_tensor_transform(ds['test'][i]['image'])
    y_loader = lambda i: ds['test'][i]['text']
    ds_test = GenericXYLambdaAutoLabelEncoderDataset(x_loader, y_loader, len(ds['test']))
    return ds_test


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
    working_dir_manager = WorkingDirManager('tmp/exp_cub_v971')
    working_dir_manager.swtich_to_exp_dir()

    # prepare dataset
    ds = load_dataset("cassiekang/cub200_dataset")
    image_item_to_tensor_transform = CommonImageToDlTensorForTraining()
    x_loader = lambda i: image_item_to_tensor_transform(ds['train'][i]['image'])
    y_loader = lambda i: ds['train'][i]['text']
    ds_train = GenericXYLambdaAutoLabelEncoderDataset(x_loader, y_loader, len(ds['train']))
    # train model
    
    
    DeepMetricLearningImageTrainverV971(ds_train, ds_train.get_number_of_classes(), model_name='resnet18', loss_name='arcface_loss', embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)

    if not AcceleratorSetting.is_local_main_process():
        # the evaluation is only done on the local main process
        return

    # evaluate model
    results_summary_of_each_epoch = []
    for epoch in range(1, ExpCubConfig.num_epochs + 1):
        model_path = f'model_epoch_{epoch:03d}.pth'
        if os.path.exists(model_path):
            results = evaluate_one_epoch(model_path)
            result_of_this_epoch = {
                'epoch': epoch,
                'top1_accuracy': results['top1_accuracy'],
                'pr_auc': results['pr_auc']
            }
            print(result_of_this_epoch)
            results_summary_of_each_epoch.append(result_of_this_epoch)

    results_summary_of_each_epoch_df = pd.DataFrame(results_summary_of_each_epoch)
    results_summary_of_each_epoch_df.to_csv('results_summary_of_each_epoch.csv', index=False)
    
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
    ds_test = get_test_dataset_with_encoded_labels()
    
    # Load model as ImageModelWrapper
    model_wrapper = Resnet18MetricModel(ExpCubConfig.embedding_dim, model_param_path=model_path)
    
    # Get embeddings using images_to_embeddings_one_by_one
    print("Running inference on test set...")
    # roughly each batch size = GPU memory / 30 MB. E.g. 20GB GPU memory, batch size = 667.
    all_embeddings = infer_x_dataset_with_simple_stacking(ds_test, model_wrapper, batch_size=100)
    
    # Create dataframe for evaluation
    embeddings_df = pd.DataFrame({
        'embedding': [emb for emb in all_embeddings],
        'label': ds_test.get_y_list_with_encoded_labels()
    })

    # Evaluate PR AUC
    print("Computing PR AUC...")
    pairwise_similarity_ground_truth_matrix = create_pairwise_similarity_ground_truth_matrix(ds_test.get_y_list_with_encoded_labels())
    pairwise_similarity_score_matrix = calculate_cosine_similarity_matrix(all_embeddings)
    metrics = evaluate_pairwise_score_matrix_with_true_label(pairwise_similarity_ground_truth_matrix, pairwise_similarity_score_matrix)

    return metrics


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
    # roughly each batch size = GPU memory / 30 MB. E.g. 20GB GPU memory, batch size = 667.
    all_embeddings = infer_x_dataset_with_simple_stacking(ds_test, model_wrapper, batch_size=100)
    
    # Create dataframe for evaluation
    embeddings_df = pd.DataFrame({
        'embedding': [emb for emb in all_embeddings],
        'label': ds_test.get_y_list_with_encoded_labels()
    })
    
    print("Computing evaluation metrics...")
    results = evaluate_embedding_top1_accuracy_ignore_self(embeddings_df)
    
    print(f"Top-1 Accuracy: {results['avg_top1_accuracy']:.4f}")
    print(f"Accuracy Upper Bound: {results['accuracy_upper_bound']:.4f}")

    pairwise_similarity_ground_truth_matrix = create_pairwise_similarity_ground_truth_matrix(ds_test.get_y_list_with_encoded_labels())

    pairwise_similarity_score_matrix = calculate_cosine_similarity_matrix(all_embeddings)

    metrics = evaluate_pairwise_score_matrix_with_true_label(pairwise_similarity_ground_truth_matrix, pairwise_similarity_score_matrix)
    print(f"PR AUC: {metrics['pr_auc']:.4f}")


def exp_run_evaluation_on_each_epoch():
    """
    Run evaluation on each epoch.
    """
    # Load dataset
    ds_test = get_test_dataset_with_encoded_labels()
    
    DeepMetricLearningImageEvaluatorOnEachEpoch(ds_test, 'resnet18', ExpCubConfig.embedding_dim, 'tmp/exp_cub_v971', ExpCubConfig.num_epochs, 'tmp/exp_cub_v971_evaluation_report')


def main():
    main_cmd_manager = MainCmdManager()
    main_cmd_manager.register_main_cmd('exp_cub_v971', exp_cub_v971)
    main_cmd_manager.register_main_cmd('exp_eval_pretrained_resenet18', exp_eval_pretrained_resenet18)
    main_cmd_manager.register_main_cmd('exp_run_evaluation_on_each_epoch', exp_run_evaluation_on_each_epoch)
    main_cmd_manager.main()

if __name__ == "__main__":
    main()