import requests
from PIL import Image
import base64
import io
import time
from typing import Union, Optional
from torchvision import transforms
from easydl.utils import smart_print


COMMON_IMAGE_PREPROCESSING_FOR_TRAINING = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

COMMON_IMAGE_PREPROCESSING_FOR_TESTING = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ImageLoadError(Exception):
    """Base exception for all image loading errors."""
    pass


class NetworkImageError(ImageLoadError):
    """Exception raised for errors loading images from URLs."""
    pass


class FileImageError(ImageLoadError):
    """Exception raised for errors loading images from files."""
    pass


class Base64ImageError(ImageLoadError):
    """Exception raised for errors decoding base64 images."""
    pass


class S3ImageError(ImageLoadError):
    """Exception raised for errors loading images from S3."""
    pass


def smart_read_image(image_str: Union[str, Image.Image], auto_retry: int = 0) -> Image.Image:
    """
    Read an image from various sources and convert to RGB format.
    
    Args:
        image_str: Source of the image. Can be:
            - PIL Image object
            - Path to an image file
            - URL (http:// or https://)
            - Local file (file://)
            - Base64 encoded image (base64://)
            - S3 path (s3://)
        auto_retry: Number of retry attempts if loading fails (default: 0)
    
    Returns:
        PIL Image in RGB format
        
    Raises:
        NetworkImageError: Error loading image from URL
        FileImageError: Error loading image from local file
        Base64ImageError: Error decoding base64 image
        S3ImageError: Error loading image from S3
        ImageLoadError: Other image loading errors
        ValueError: Invalid input type or format
    """
    if auto_retry > 0:
        last_exception = None
        for attempt in range(auto_retry):
            try:
                return smart_read_image(image_str, 0)  # Call with no retry to avoid recursion
            except Exception as e:
                last_exception = e
                smart_print(f"Error reading image (attempt {attempt+1}/{auto_retry}): {e}")
                time.sleep(0.1 * (attempt + 1))  # Increasing backoff
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise ImageLoadError(f"Failed after {auto_retry} attempts. Last error: {last_exception}")

    # Return image if it is already an image
    if isinstance(image_str, Image.Image):
        return image_str.convert('RGB')
    
    # Validate input type
    if not isinstance(image_str, str):
        raise ValueError(f"Expected string or PIL.Image, got {type(image_str).__name__}")
    
    try:
        if image_str.startswith(('http://', 'https://')):
            try:
                response = requests.get(image_str, stream=True, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                image = Image.open(response.raw)
            except requests.RequestException as e:
                raise NetworkImageError(f"Failed to fetch image from URL: {e}")
            except Exception as e:
                raise NetworkImageError(f"Error processing image from URL: {e}")
                
        elif image_str.startswith('file://'):
            try:
                filepath = image_str.split('file://', 1)[1]
                image = Image.open(filepath)
            except (FileNotFoundError, PermissionError) as e:
                raise FileImageError(f"Failed to access file: {e}")
            except Exception as e:
                raise FileImageError(f"Error opening image file: {e}")
                
        elif image_str.startswith('base64://'):
            try:
                base64_data = image_str.split('base64://', 1)[1]
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
            except base64.binascii.Error as e:
                raise Base64ImageError(f"Invalid base64 encoding: {e}")
            except Exception as e:
                raise Base64ImageError(f"Error decoding base64 image: {e}")
                
        elif image_str.startswith('s3://'):
            try:
                import boto3
                s3 = boto3.client('s3')
                parts = image_str.split('s3://', 1)[1].split('/', 1)
                
                if len(parts) < 2:
                    raise S3ImageError(f"Invalid S3 path format: {image_str}. Expected s3://bucket/key")
                    
                bucket_name = parts[0]
                object_key = parts[1]
                
                try:
                    response = s3.get_object(Bucket=bucket_name, Key=object_key)
                    image = Image.open(io.BytesIO(response['Body'].read()))
                except s3.exceptions.NoSuchBucket:
                    raise S3ImageError(f"S3 bucket does not exist: {bucket_name}")
                except s3.exceptions.NoSuchKey:
                    raise S3ImageError(f"S3 object does not exist: {object_key}")
                except Exception as e:
                    raise S3ImageError(f"Error accessing S3 object: {e}")
                    
            except ImportError:
                raise ImportError("boto3 is required for S3 support. Install with 'pip install boto3'")
                
        else:
            # Default to local file path
            try:
                image = Image.open(image_str)
            except (FileNotFoundError, PermissionError) as e:
                raise FileImageError(f"Failed to access file: {e}")
            except Exception as e:
                raise FileImageError(f"Error opening image file: {e}")
                
        # Convert to RGB format
        image = image.convert('RGB')
        return image
        
    except (NetworkImageError, FileImageError, Base64ImageError, S3ImageError) as e:
        # Re-raise specific exceptions for easier catching
        raise
        
    except Exception as e:
        # Catch any other unexpected errors
        raise ImageLoadError(f"Unexpected error loading image: {e}")


def smart_read_and_crop_image_by_box(image_str: Union[str, Image.Image], box: tuple, auto_retry: int = 0) -> Image.Image:
    """
    Read an image from various sources, crop it by a bounding box, and return the cropped image.
    
    Args:
        image_str: Source of the image (PIL Image, path, URL, file://, base64://, s3://)
        box: Tuple of (left, upper, right, lower) coordinates for cropping
        auto_retry: Number of retry attempts if loading fails (default: 0)
    
    Returns:
        Cropped PIL Image in RGB format
        
    Raises:
        Same exceptions as smart_read_image()
        ValueError: If box coordinates are invalid
    """
    image = smart_read_image(image_str, auto_retry)
    
    # Handle invalid box coordinates (right <= left or bottom <= top)
    left, upper, right, lower = box
    # Ensure right > left
    if right <= left:
        # Create empty width image with correct height
        return Image.new('RGB', (0, max(0, lower - upper)), color='black')
    # Ensure bottom > top
    if lower <= upper:
        # Create empty height image with correct width
        return Image.new('RGB', (max(0, right - left), 0), color='black')
        
    return image.crop(box)


class ImageToDlTensor:
    def __init__(self, image_preprocessing_function):
        self.image_preprocessing_function = image_preprocessing_function

    def __call__(self, image):
        image = smart_read_image(image)
        return self.image_preprocessing_function(image)

class CommonImageToDlTensorForTraining(ImageToDlTensor):
    def __init__(self):
        super().__init__(COMMON_IMAGE_PREPROCESSING_FOR_TRAINING)

class CommonImageToDlTensorForTesting(ImageToDlTensor):
    def __init__(self):
        super().__init__(COMMON_IMAGE_PREPROCESSING_FOR_TESTING)

