
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier
# Test that Resnet18MetricModel can be loaded with pretrained weights
from easydl.dml.pytorch_models import Resnet18MetricModel
import torch

def test_imagenet_resnet18_classifier():
    classifier = create_imagenet_resnet18_classifier()
    image = 'https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg'
    label, score = classifier.predict_label_with_confidence(image)
    print(f"label: {label}, score: {score}")
    assert label == "beagle"
    assert score > 0.1



def test_resnet18_metric_model_pretrained_loads():
    model = Resnet18MetricModel(embedding_dim=32)
    # Check that the model is a torch.nn.Module and has .backbone/.embedding
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'embedding')
    assert callable(getattr(model, 'forward', None))
    # Check the embedding layer dimensions
    assert model.embedding.out_features == 32
    # Forward a random input, output should have normalized shape
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 32)
    # Output should be normalized (l2 close to 1.0)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


