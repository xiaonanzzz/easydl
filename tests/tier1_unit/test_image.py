"""
Tier 1 Unit Tests: Image Loading and Processing

Fast tests for image utilities with mocking.
"""

import base64
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from easydl.image import (
    Base64ImageError,
    FileImageError,
    ImageLoadError,
    NetworkImageError,
    S3ImageError,
    smart_read_and_crop_image_by_box,
    smart_read_image,
)


@pytest.mark.unit
class TestSmartReadImage:
    """Unit tests for smart_read_image function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = Image.new("RGB", (100, 100), color="red")
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_file"):
            os.unlink(self.temp_file.name)

    def test_read_pil_image(self):
        """Test reading from a PIL Image object."""
        result = smart_read_image(self.test_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_read_local_file(self):
        """Test reading from a local file path."""
        result = smart_read_image(self.temp_file.name)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_read_base64(self):
        """Test reading from a base64 encoded string."""
        buffer = io.BytesIO()
        self.test_image.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_url = f"base64://{base64_data}"

        result = smart_read_image(base64_url)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @patch("requests.get")
    def test_read_http_url(self, mock_get):
        """Test reading from an HTTP URL with mocking."""
        mock_response = MagicMock()
        mock_response.raw = io.BytesIO()
        self.test_image.save(mock_response.raw, format="PNG")
        mock_response.raw.seek(0)
        mock_get.return_value = mock_response

        result = smart_read_image("http://example.com/image.png")

        mock_get.assert_called_once()
        assert isinstance(result, Image.Image)

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(ValueError):
            smart_read_image(123)

    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(FileImageError):
            smart_read_image("/nonexistent/path/to/image.jpg")


@pytest.mark.unit
class TestSmartReadAndCropImageByBox:
    """Unit tests for smart_read_and_crop_image_by_box function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = Image.new("RGB", (100, 100), color="red")
        blue_square = Image.new("RGB", (40, 40), color="blue")
        self.test_image.paste(blue_square, (30, 30))

    def test_crop_pil_image(self):
        """Test cropping a PIL Image object."""
        box = (30, 30, 70, 70)
        result = smart_read_and_crop_image_by_box(self.test_image, box)

        assert isinstance(result, Image.Image)
        assert result.size == (40, 40)

    def test_crop_returns_correct_region(self):
        """Test that cropping returns the correct region."""
        box = (30, 30, 70, 70)
        result = smart_read_and_crop_image_by_box(self.test_image, box)

        # Center pixel should be blue
        pixel = result.getpixel((20, 20))
        assert pixel[2] > 200  # Blue channel should be high


@pytest.mark.unit
class TestImageErrorHandling:
    """Unit tests for image error handling."""

    def test_file_image_error(self):
        """Test FileImageError is raised for nonexistent files."""
        with pytest.raises(FileImageError):
            smart_read_image("/nonexistent/path/to/image.jpg")

    def test_base64_image_error(self):
        """Test Base64ImageError is raised for invalid base64."""
        with pytest.raises(Base64ImageError):
            smart_read_image("base64://invalid-base64-data")

    @patch("requests.get")
    def test_network_image_error(self, mock_get):
        """Test NetworkImageError is raised for network failures."""
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(NetworkImageError):
            smart_read_image("http://example.com/image.jpg")
