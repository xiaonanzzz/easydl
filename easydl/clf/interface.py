"""
Interface definitions for classification models.
"""


class ImagenetClassifierInterface:
    """
    Interface for ImageNet classifiers.

    Defines the contract for ImageNet classification implementations.
    Subclasses must implement predict_label and predict_label_with_confidence.
    """

    def predict_label(self, image) -> str:
        """
        Predict the ImageNet label for an image.

        Args:
            image: Image input (file path, URL, PIL Image, or base64 string).

        Returns:
            str: The predicted ImageNet class label.
        """
        raise NotImplementedError

    def predict_label_with_confidence(self, image) -> tuple:
        """
        Predict the ImageNet label and confidence score for an image.

        Args:
            image: Image input (file path, URL, PIL Image, or base64 string).

        Returns:
            tuple: (label, score) where label is the predicted class name
                and score is the confidence probability (0-1).
        """
        raise NotImplementedError
