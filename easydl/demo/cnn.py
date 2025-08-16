import torch
import torch.nn.functional as F
import numpy as np

def generate_pairwise_similiarty_heatmap(feature_map_1, feature_map_2):
    """
    Optimized version using vectorized operations.
    
    Args:
        feature_map_1: (H, W, C) tensor
        feature_map_2: (H, W, C) tensor
        
    Returns:
        similarity_map_1: (H*W, H*W) tensor
        similarity_map_2: (H*W, H*W) tensor
    """
    if isinstance(feature_map_1, np.ndarray):
        feature_map_1 = torch.from_numpy(feature_map_1)
    if isinstance(feature_map_2, np.ndarray):
        feature_map_2 = torch.from_numpy(feature_map_2)
    
    # Flatten the feature maps: (H, W, C) -> (H*W, C)
    H1, W1, C1 = feature_map_1.shape
    H2, W2, C2 = feature_map_2.shape
    if C1 != C2:
        raise ValueError("Feature map hidden dimension must match")
    
    feature_map_1_flat = feature_map_1.view(H1*W1, C1)  # (H*W, C)
    feature_map_2_flat = feature_map_2.view(H2*W2, C2)  # (H*W, C)
    
    # Normalize feature vectors for cosine similarity
    feature_map_1_norm = F.normalize(feature_map_1_flat, p=2, dim=1)  # (H*W, C)
    feature_map_2_norm = F.normalize(feature_map_2_flat, p=2, dim=1)  # (H*W, C)
    
    # Compute cosine similarity matrix: (H*W, H*W)
    # This is equivalent to: feature_map_1_norm @ feature_map_2_norm.T
    similarity_matrix = torch.mm(feature_map_1_norm, feature_map_2_norm.T)
    similarity_matrix = similarity_matrix.reshape(H1, W1, H2, W2)
    similarity_1 = similarity_matrix.mean(dim=(2, 3))
    similarity_2 = similarity_matrix.mean(dim=(0, 1))
    return similarity_1, similarity_2

import numpy as np
from PIL import Image, ImageEnhance

def get_jet_color(value):
    """
    A simple function to manually simulate a 'jet' colormap.
    The input value is expected to be normalized between 0 and 1.
    """
    if value < 0.34:  # Blue to cyan
        return (0, int(value * 3 * 255), 255)
    elif value < 0.67:  # Cyan to yellow
        return (int((value - 0.34) * 3 * 255), 255, 255 - int((value - 0.34) * 3 * 255))
    else:  # Yellow to red
        return (255, 255 - int((value - 0.67) * 3 * 255), 0)

def overlay_heatmap(image, heatmap_matrix):
    """
    Overlays a heatmap matrix onto a color image using PIL.

    This function opens an image, normalizes and resizes the heatmap,
    applies a simulated 'jet' colormap, and blends it with the original image.

    Args:
        image_path (str): The file path to the input image (e.g., 'image.jpg').
        heatmap_matrix (np.ndarray): A 2D NumPy array representing the heatmap.
                                     The values can be floats or integers.

    Returns:
        PIL.Image.Image: The blended PIL Image object with the heatmap overlay.
                         Returns None if the image cannot be loaded.
    """

    # Get the dimensions of the image
    w, h = image.size

    # Normalize the heatmap matrix to a range of [0, 1]
    # This is necessary for our manual color mapping.
    if isinstance(heatmap_matrix, torch.Tensor):
        heatmap_matrix = heatmap_matrix.cpu().numpy()
    min_val = np.min(heatmap_matrix)
    max_val = np.max(heatmap_matrix)
    if (max_val - min_val) == 0:
        normalized_heatmap = np.zeros_like(heatmap_matrix, dtype=np.float32)
    else:
        normalized_heatmap = (heatmap_matrix - min_val) / (max_val - min_val)

    # Resize the normalized heatmap to match the image dimensions
    heatmap_pil = Image.fromarray((normalized_heatmap * 255).astype(np.uint8), 'L')
    heatmap_resized = heatmap_pil.resize((w, h), Image.LANCZOS)

    # Apply the manual 'jet' colormap to the resized heatmap
    heatmap_colored_data = [get_jet_color(val / 255.0) for val in heatmap_resized.getdata()]
    heatmap_colored = Image.new('RGB', (w, h))
    heatmap_colored.putdata(heatmap_colored_data)

    # Blend the heatmap with the original image
    # The alpha value (0.5) controls the transparency of the overlay
    blended_image = Image.blend(image, heatmap_colored, alpha=0.5)

    return blended_image
