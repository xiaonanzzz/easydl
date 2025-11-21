
from easydl.clf.pytorch_models import create_imagenet_resnet18_classifier


def test_imagenet_resnet18_classifier():
    classifier = create_imagenet_resnet18_classifier()
    image = 'https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg'
    label, score = classifier.predict_label_with_confidence(image)
    print(f"label: {label}, score: {score}")
    assert label == "beagle"
    assert score > 0.1





