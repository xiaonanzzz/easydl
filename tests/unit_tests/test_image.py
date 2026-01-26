import base64
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from easydl.image import (
    Base64ImageError,
    CommonImageToDlTensorForTesting,
    CommonImageToDlTensorForTraining,
    FileImageError,
    ImageLoadError,
    ImageToDlTensor,
    NetworkImageError,
    S3ImageError,
    smart_read_and_crop_image_by_box,
    smart_read_image,
)


class TestSmartReadImage:

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create a simple test image
        self.test_image = Image.new("RGB", (100, 100), color="red")

        # Create a temporary file for file-based tests
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()

    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        # Remove temporary file
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

    def test_read_file_url(self):
        """Test reading from a file:// URL."""
        file_url = f"file://{self.temp_file.name}"
        result = smart_read_image(file_url)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_read_base64(self):
        """Test reading from a base64 encoded string."""
        # Convert image to base64
        buffer = io.BytesIO()
        self.test_image.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_url = f"base64://{base64_data}"

        result = smart_read_image(base64_url)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    @patch("requests.get")
    def test_read_http_url(self, mock_get):
        """Test reading from an HTTP URL."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.raw = io.BytesIO()
        self.test_image.save(mock_response.raw, format="PNG")
        mock_response.raw.seek(0)  # Reset file pointer after writing
        mock_get.return_value = mock_response

        result = smart_read_image("http://example.com/image.png")

        # Check that requests.get was called correctly
        mock_get.assert_called_once_with(
            "http://example.com/image.png", stream=True, timeout=10
        )
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @patch("boto3.client")
    def test_read_s3_url(self, mock_boto_client):
        """Test reading from an S3 URL."""
        # Mock the S3 client and response
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        mock_response = {"Body": MagicMock()}
        buffer = io.BytesIO()
        self.test_image.save(buffer, format="PNG")
        buffer.seek(0)
        mock_response["Body"].read.return_value = buffer.getvalue()
        mock_s3.get_object.return_value = mock_response

        result = smart_read_image("s3://bucket/path/to/image.png")

        # Check S3 client was used correctly
        mock_boto_client.assert_called_once_with("s3")
        mock_s3.get_object.assert_called_once_with(
            Bucket="bucket", Key="path/to/image.png"
        )
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_auto_retry(self):
        """Test auto retry functionality."""
        with patch(
            "easydl.image.smart_read_image",
            side_effect=[ValueError("Test error"), self.test_image],
        ) as mock_read:
            # The first call raises an error, the second returns the image
            # We patch the same function we're testing but that's ok because we're testing the retry wrapper
            result = smart_read_image("dummy", auto_retry=2)

            assert isinstance(result, Image.Image)
            assert result.mode == "RGB"
            assert mock_read.call_count == 2

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(ValueError):
            smart_read_image(123)  # Not a string or PIL.Image

    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(FileImageError):
            smart_read_image("/nonexistent/path/to/image.jpg")


class TestSmartReadAndCropImageByBox:

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create a test image with different colored regions
        self.test_image = Image.new("RGB", (100, 100), color="red")
        # Draw a blue square in the middle
        blue_square = Image.new("RGB", (40, 40), color="blue")
        self.test_image.paste(blue_square, (30, 30))

        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()

    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        # Remove temporary file
        if hasattr(self, "temp_file"):
            os.unlink(self.temp_file.name)

    def test_crop_pil_image(self):
        """Test cropping a PIL Image object."""
        box = (30, 30, 70, 70)  # Coordinates for blue square
        result = smart_read_and_crop_image_by_box(self.test_image, box)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (40, 40)
        # Check pixel color of the cropped image (should be blue)
        pixel = result.getpixel((20, 20))
        assert pixel == (0, 0, 255) or pixel[2] > 200  # Blue has high B value

    def test_crop_local_file(self):
        """Test cropping from a local file path."""
        box = (30, 30, 70, 70)  # Coordinates for blue square
        result = smart_read_and_crop_image_by_box(self.temp_file.name, box)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (40, 40)
        # Check pixel color of the cropped image (should be blue)
        pixel = result.getpixel((20, 20))
        assert pixel == (0, 0, 255) or pixel[2] > 200  # Blue has high B value

    @patch("easydl.image.smart_read_image")
    def test_smart_read_is_called(self, mock_smart_read):
        """Test that smart_read_image is called with correct parameters."""
        mock_smart_read.return_value = self.test_image
        box = (10, 20, 30, 40)

        smart_read_and_crop_image_by_box("test_path", box, auto_retry=3)

        # Verify smart_read_image was called with correct parameters
        mock_smart_read.assert_called_once_with("test_path", 3)

    def test_invalid_box_values(self):
        """Test with invalid box values."""
        # Boxes where right <= left or bottom <= top should still work but result in empty image
        box = (50, 50, 30, 70)  # right < left
        result = smart_read_and_crop_image_by_box(self.test_image, box)
        assert result.size == (0, 20)  # Width will be 0

        box = (30, 50, 70, 30)  # bottom < top
        result = smart_read_and_crop_image_by_box(self.test_image, box)
        assert result.size == (40, 0)  # Height will be 0


class TestImageTransformationClasses:

    def setup_method(self):
        # Create a simple test image
        self.test_image = Image.new("RGB", (100, 100), color="red")

    def test_image_to_dl_tensor_with_pil(self):
        """Test ImageToDlTensor with a PIL image input."""
        # Create a simple transformation function
        transform_fn = lambda img: img.resize((50, 50))
        transformer = ImageToDlTensor(transform_fn)

        # Apply transformation to PIL image
        result = transformer(self.test_image)

        # Check that transform function was applied
        assert result.size == (50, 50)

    def test_image_to_dl_tensor_with_path(self):
        """Test ImageToDlTensor with a file path input."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            self.test_image.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Create a simple transformation function
            transform_fn = lambda img: img.resize((50, 50))
            transformer = ImageToDlTensor(transform_fn)

            # Apply transformation using file path
            result = transformer(temp_file_path)

            # Check that transform function was applied
            assert result.size == (50, 50)
        finally:
            os.unlink(temp_file_path)

    @patch("easydl.image.COMMON_IMAGE_PREPROCESSING_FOR_TRAINING")
    def test_common_image_to_dl_tensor_for_training(self, mock_transform):
        """Test CommonImageToDlTensorForTraining class."""
        # Mock the transformation function
        mock_transform.return_value = "transformed_tensor"

        transformer = CommonImageToDlTensorForTraining()
        with patch(
            "easydl.image.smart_read_image", return_value=self.test_image
        ) as mock_read:
            result = transformer("test_image_path")

            # Check that smart_read_image was called
            mock_read.assert_called_once_with("test_image_path")
            # Check that transform was applied
            mock_transform.assert_called_once()
            assert result == "transformed_tensor"

    @patch("easydl.image.COMMON_IMAGE_PREPROCESSING_FOR_TESTING")
    def test_common_image_to_dl_tensor_for_testing(self, mock_transform):
        """Test CommonImageToDlTensorForTesting class."""
        # Mock the transformation function
        mock_transform.return_value = "transformed_tensor"

        transformer = CommonImageToDlTensorForTesting()
        with patch(
            "easydl.image.smart_read_image", return_value=self.test_image
        ) as mock_read:
            result = transformer("test_image_path")

            # Check that smart_read_image was called
            mock_read.assert_called_once_with("test_image_path")
            # Check that transform was applied
            mock_transform.assert_called_once()
            assert result == "transformed_tensor"


class TestImageErrorHandling:

    def test_network_image_error(self):
        """Test NetworkImageError is raised correctly."""
        with patch("requests.get") as mock_get:
            # Mock a network error
            mock_get.side_effect = requests.RequestException("Network error")

            with pytest.raises(NetworkImageError) as exc_info:
                smart_read_image("http://example.com/image.jpg")

            assert "Failed to fetch image from URL" in str(exc_info.value)

    def test_file_image_error(self):
        """Test FileImageError is raised correctly."""
        with pytest.raises(FileImageError) as exc_info:
            smart_read_image("/nonexistent/path/to/image.jpg")

        assert "Failed to access file" in str(exc_info.value)

    def test_base64_image_error(self):
        """Test Base64ImageError is raised correctly."""
        # Invalid base64 data
        with pytest.raises(Base64ImageError) as exc_info:
            smart_read_image("base64://invalid-base64-data")

        assert "Invalid base64 encoding" in str(exc_info.value)

    @patch("boto3.client")
    def test_s3_image_error(self, mock_boto_client):
        """Test S3ImageError is raised correctly."""
        # Mock the S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Create exceptions attribute for mock_s3
        mock_s3.exceptions = MagicMock()
        mock_s3.exceptions.NoSuchBucket = Exception

        # Mock an S3 bucket not found error
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchBucket()

        with pytest.raises(S3ImageError) as exc_info:
            smart_read_image("s3://nonexistent-bucket/image.jpg")

        assert "S3 bucket does not exist" in str(exc_info.value)

    def test_invalid_s3_path(self):
        """Test error for invalid S3 path format."""
        with patch("boto3.client"):
            with pytest.raises(S3ImageError) as exc_info:
                smart_read_image("s3://bucket-only")

            assert "Invalid S3 path format" in str(exc_info.value)

    def test_auto_retry_exhausted(self):
        """Test error when auto_retry attempts are exhausted."""
        with patch(
            "easydl.image.smart_read_image", side_effect=ValueError("Test error")
        ):
            with pytest.raises(ImageLoadError) as exc_info:
                # Call the function directly with auto_retry to avoid infinite recursion in testing
                smart_read_image("dummy", auto_retry=3)

            assert "Failed after 3 attempts" in str(exc_info.value)
