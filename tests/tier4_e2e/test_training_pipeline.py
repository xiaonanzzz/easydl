"""
Tier 4 End-to-End Tests: Full Training Pipeline

Full pipeline tests with training and evaluation.
These tests may take several minutes to complete.
"""

import os
from pathlib import Path

import pytest
import torch

from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.trainer import DeepMetricLearningImageTrainverV971
from easydl.image import (
    COMMON_IMAGE_PREPROCESSING_FOR_TESTING,
    COMMON_IMAGE_PREPROCESSING_FOR_TRAINING,
)


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingPipeline:
    """End-to-end tests for full training pipeline."""

    def test_training_completes_without_error(self, cub_train_small, exp_dir):
        """Test that training completes without errors."""
        ds_train = cub_train_small
        ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})

        original_dir = os.getcwd()
        os.chdir(exp_dir)

        try:
            DeepMetricLearningImageTrainverV971(
                ds_train,
                number_of_classes=ds_train.get_number_of_classes(),
                model_name="resnet18",
                loss_name="proxy_anchor_loss",
                embedding_dim=128,
                batch_size=16,
                num_epochs=2,
                lr=1e-4,
            )
        finally:
            os.chdir(original_dir)

    def test_model_checkpoints_saved(self, cub_train_small, exp_dir):
        """Test that model checkpoints are saved during training."""
        ds_train = cub_train_small
        ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})

        original_dir = os.getcwd()
        os.chdir(exp_dir)

        num_epochs = 2

        try:
            DeepMetricLearningImageTrainverV971(
                ds_train,
                number_of_classes=ds_train.get_number_of_classes(),
                model_name="resnet18",
                loss_name="proxy_anchor_loss",
                embedding_dim=128,
                batch_size=16,
                num_epochs=num_epochs,
                lr=1e-4,
            )

            # Check that checkpoints exist
            for epoch in range(1, num_epochs + 1):
                checkpoint = exp_dir / f"model_epoch_{epoch:03d}.pth"
                assert checkpoint.exists(), f"Missing checkpoint for epoch {epoch}"
        finally:
            os.chdir(original_dir)

    def test_saved_model_loadable(self, cub_train_small, exp_dir):
        """Test that saved model can be loaded and used."""
        ds_train = cub_train_small
        ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})

        original_dir = os.getcwd()
        os.chdir(exp_dir)

        try:
            DeepMetricLearningImageTrainverV971(
                ds_train,
                number_of_classes=ds_train.get_number_of_classes(),
                model_name="resnet18",
                loss_name="proxy_anchor_loss",
                embedding_dim=128,
                batch_size=16,
                num_epochs=1,
                lr=1e-4,
            )

            # Load saved model
            checkpoint = exp_dir / "model_epoch_001.pth"
            assert checkpoint.exists()

            model = Resnet18MetricModel(embedding_dim=128)
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            model.eval()

            # Verify model works
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)

            assert output.shape == (1, 128)
        finally:
            os.chdir(original_dir)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu
class TestTrainingOnGPU:
    """End-to-end GPU training tests."""

    def test_gpu_training(self, cub_train_small, exp_dir, device):
        """Test training on GPU."""
        if device.type != "cuda":
            pytest.skip("GPU not available")

        ds_train = cub_train_small
        ds_train.extend_lambda_dict({"x": COMMON_IMAGE_PREPROCESSING_FOR_TRAINING})

        original_dir = os.getcwd()
        os.chdir(exp_dir)

        try:
            DeepMetricLearningImageTrainverV971(
                ds_train,
                number_of_classes=ds_train.get_number_of_classes(),
                model_name="resnet18",
                loss_name="proxy_anchor_loss",
                embedding_dim=128,
                batch_size=16,
                num_epochs=1,
                lr=1e-4,
            )
        finally:
            os.chdir(original_dir)
