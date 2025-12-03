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
from easydl.image import COMMON_IMAGE_PREPROCESSING_FOR_TESTING, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING
import os
from pathlib import Path
from easydl.common_infer import infer_x_dataset_with_simple_stacking
from easydl.utils import AcceleratorSetting
from sklearn.metrics.pairwise import cosine_similarity
from easydl.dml.evaluation import create_pairwise_similarity_ground_truth_matrix, evaluate_pairwise_score_matrix_with_true_label, calculate_cosine_similarity_matrix
from easydl.utils import WorkingDirManager, MainCmdManager
from easydl.dml.evaluation import DeepMetricLearningImageEvaluatorOnEachEpoch
from easydl.dml.evaluation import StandardEmbeddingEvaluationV1
from easydl.public_dataset.cub import get_train_dataset_with_image_and_encoded_labels, get_test_dataset_with_image_and_encoded_labels, get_small_train_dataset_with_image_and_encoded_labels

class ExpCubConfig:
    embedding_dim = 384
    batch_size = 256
    num_epochs = 100
    lr = 1e-4

def get_data_augmentation_transform():
    import torchvision.transforms as transforms
    from timm.data import rand_augment_transform
    data_aug_tfm = rand_augment_transform('rand-m5-n2')
    train_transforms = transforms.Compose([
        data_aug_tfm,
        transforms.Resize(256), 
        transforms.RandomCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms

def do_dml_experiment_with_cub_dataset():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--quick_mode', action='store_true', default=False)
    args_parser.add_argument('--model_name', type=str, default='resnet18')
    args_parser.add_argument('--loss_name', type=str, default='proxy_anchor_loss')
    args_parser.add_argument('--exp_dir', type=str, default='tmp/exp_cub_972')

    args, _ = args_parser.parse_known_args()
    quick_mode = args.quick_mode

    # prepare output directory
    working_dir_manager = WorkingDirManager(args.exp_dir)
    working_dir_manager.swtich_to_exp_dir()

    if quick_mode:
        ds_train = get_small_train_dataset_with_image_and_encoded_labels(num_samples=200)
        ds_test = get_small_train_dataset_with_image_and_encoded_labels(num_samples=200)
        ExpCubConfig.num_epochs = 2
    else:
        ds_train = get_train_dataset_with_image_and_encoded_labels()
        ds_test = get_test_dataset_with_image_and_encoded_labels()

    ds_train.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})
    ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

    # run training
    DeepMetricLearningImageTrainverV971(ds_train, ds_train.get_number_of_classes(), model_name=args.model_name, loss_name=args.loss_name, embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)

    if not AcceleratorSetting.is_local_main_process():
        # the evaluation is only done on the local main process
        return

    working_dir_manager.switch_back_to_original_working_dir()

    evaluator = DeepMetricLearningImageEvaluatorOnEachEpoch(ds_test, args.model_name, ExpCubConfig.embedding_dim, args.exp_dir, 
    ExpCubConfig.num_epochs, args.exp_dir)

    print('Evaluation report location: ', evaluator.report_csv_path)
    df = pd.read_csv(evaluator.report_csv_path)
    print(df)
    
def do_dml_experiment_with_cub_dataset_with_data_augmentation():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--quick_mode', action='store_true', default=False)
    args_parser.add_argument('--model_name', type=str, default='resnet18')
    args_parser.add_argument('--loss_name', type=str, default='proxy_anchor_loss')
    args_parser.add_argument('--exp_dir', type=str, default='tmp/exp_cub_972')

    args, _ = args_parser.parse_known_args()
    quick_mode = args.quick_mode

    # prepare output directory
    working_dir_manager = WorkingDirManager(args.exp_dir)
    working_dir_manager.swtich_to_exp_dir()

    if quick_mode:
        ds_train = get_small_train_dataset_with_image_and_encoded_labels(num_samples=200)
        ds_test = get_small_train_dataset_with_image_and_encoded_labels(num_samples=200)
        ExpCubConfig.num_epochs = 2
    else:
        ds_train = get_train_dataset_with_image_and_encoded_labels()
        ds_test = get_test_dataset_with_image_and_encoded_labels()

    ds_train.extend_lambda_dict({'x': get_data_augmentation_transform()})
    ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})

    # run training
    DeepMetricLearningImageTrainverV971(ds_train, ds_train.get_number_of_classes(), model_name=args.model_name, loss_name=args.loss_name, embedding_dim=ExpCubConfig.embedding_dim, batch_size=ExpCubConfig.batch_size, num_epochs=ExpCubConfig.num_epochs, lr=ExpCubConfig.lr)

    if not AcceleratorSetting.is_local_main_process():
        # the evaluation is only done on the local main process
        return

    working_dir_manager.switch_back_to_original_working_dir()

    evaluator = DeepMetricLearningImageEvaluatorOnEachEpoch(ds_test, args.model_name, ExpCubConfig.embedding_dim, args.exp_dir, 
    ExpCubConfig.num_epochs, args.exp_dir)

    print('Evaluation report location: ', evaluator.report_csv_path)
    df = pd.read_csv(evaluator.report_csv_path)
    print(df)

def exp_eval_a_model_on_cub_test_set(embedding_model, image_to_tensor_transform):
    ds_test = get_test_dataset_with_image_and_encoded_labels()
    ds_test.extend_lambda_dict({'x': image_to_tensor_transform})
    metrics = StandardEmbeddingEvaluationV1.evaluate_given_dataset(ds_test, embedding_model)
    return metrics

def exp_eval_pretrained_resenet18():
    tensor_to_embedding_model = Resnet18MetricModel(128)
    image_to_tensor_transform = COMMON_IMAGE_PREPROCESSING_FOR_TESTING
    metrics = exp_eval_a_model_on_cub_test_set(tensor_to_embedding_model, image_to_tensor_transform)
    print(metrics)


def exp_run_evaluation_on_each_epoch():
    """
    Run evaluation on each epoch.
    """
    # Load dataset
    ds_test = get_test_dataset_with_image_and_encoded_labels()
    ds_test.extend_lambda_dict({'x': COMMON_IMAGE_PREPROCESSING_FOR_TESTING})
    DeepMetricLearningImageEvaluatorOnEachEpoch(ds_test, 'resnet18', ExpCubConfig.embedding_dim, 'tmp/exp_cub_v971', ExpCubConfig.num_epochs, 'tmp/exp_cub_v971_evaluation_report')


def main():
    main_cmd_manager = MainCmdManager()
    main_cmd_manager.register_main_cmd('do_dml_experiment_with_cub_dataset', do_dml_experiment_with_cub_dataset)
    main_cmd_manager.register_main_cmd('exp_eval_pretrained_resenet18', exp_eval_pretrained_resenet18)
    main_cmd_manager.register_main_cmd('exp_run_evaluation_on_each_epoch', exp_run_evaluation_on_each_epoch)
    main_cmd_manager.register_main_cmd('do_dml_experiment_with_cub_dataset_with_data_augmentation', do_dml_experiment_with_cub_dataset_with_data_augmentation)
    main_cmd_manager.main()

if __name__ == "__main__":
    main()