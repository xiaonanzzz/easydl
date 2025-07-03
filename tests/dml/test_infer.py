import numpy as np
from easydl.dml.infer import images_to_embeddings

class DummyModel:
    def __call__(self, image):
        # return a constant vector for simplicity
        return np.array([1.0])

def test_images_to_embeddings_custom_reader():
    called = []
    def reader(x):
        called.append(x)
        return x  # pass through
    model = DummyModel()
    images = ['a', 'b', 'c']
    result = images_to_embeddings(images, model, image_reader=reader)
    # ensure custom reader was used on each image
    assert called == images
    assert result.shape == (3, 1)
