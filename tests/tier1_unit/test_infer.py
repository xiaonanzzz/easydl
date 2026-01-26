"""
Tier 1 Unit Tests: Inference Utilities

Fast tests for inference functions with mocking.
"""

import numpy as np
import pytest

from easydl.dml.infer import images_to_embeddings


class DummyModel:
    """Dummy model for testing."""

    def __call__(self, image):
        return np.array([1.0, 2.0, 3.0])


@pytest.mark.unit
class TestImagesToEmbeddings:
    """Unit tests for images_to_embeddings function."""

    def test_custom_reader(self):
        """Test that custom image reader is called."""
        called = []

        def reader(x):
            called.append(x)
            return x

        model = DummyModel()
        images = ["a", "b", "c"]
        result = images_to_embeddings(images, model, image_reader=reader)

        assert called == images
        assert result.shape == (3, 3)

    def test_output_shape(self):
        """Test that output shape matches number of images."""
        model = DummyModel()
        images = ["img1", "img2", "img3", "img4"]
        result = images_to_embeddings(images, model, image_reader=lambda x: x)

        assert result.shape[0] == len(images)

    def test_embeddings_stacked_correctly(self):
        """Test that embeddings are stacked into correct array."""

        class SequentialModel:
            def __init__(self):
                self.counter = 0

            def __call__(self, image):
                self.counter += 1
                return np.array([self.counter, self.counter * 2])

        model = SequentialModel()
        images = ["a", "b", "c"]
        result = images_to_embeddings(images, model, image_reader=lambda x: x)

        expected = np.array([[1, 2], [2, 4], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_image_list(self):
        """Test handling of empty image list."""
        model = DummyModel()
        images = []
        result = images_to_embeddings(images, model, image_reader=lambda x: x)

        assert result.shape[0] == 0
