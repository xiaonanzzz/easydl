import requests
from PIL import Image
import base64
import io
import time
from torchvision import transforms


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

def smart_read_image(image_str: str, auto_retry=0) -> Image.Image:
    """
    Read an image from a string, support all these formats:
    - Path to an image file (default)
    - http://...
    - https://...
    - file://...
    - base64://...

    set auto_retry to a positive integer to retry reading the image if it fails.
    """
    if auto_retry > 0:
        for _ in range(auto_retry):
            try:
                return smart_read_image(image_str)
            except Exception as e:
                print(f"Error reading image: {e}")
                time.sleep(0.1)

    # return image if it is already an image. This is why it is smart!!!
    if isinstance(image_str, Image.Image):
        return image_str.convert('RGB')

    if image_str.startswith('http'):
        image = Image.open(requests.get(image_str, stream=True).raw)
    elif image_str.startswith('file://'):
        image = Image.open(image_str.split('file://')[1])
    elif image_str.startswith('base64://'):
        image = Image.open(io.BytesIO(base64.b64decode(image_str.split('base64://')[1])))
    elif image_str.startswith('s3://'):
        import boto3
        s3 = boto3.client('s3')
        bucket_name = image_str.split('s3://')[1].split('/')[0]
        object_key = '/'.join(image_str.split('s3://')[1].split('/')[1:])
        image = Image.open(io.BytesIO(s3.get_object(Bucket=bucket_name, Key=object_key)['Body'].read()))
    else:
        # default to be a file path
        image = Image.open(image_str)

    image = image.convert('RGB')
    return image

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