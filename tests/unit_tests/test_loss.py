import torch
import torch.nn.functional as F

from easydl.dml.loss import ArcFace


def test_arcface_basic_forward():
    """Test basic forward pass of ArcFace loss"""
    batch_size = 4
    embedding_dim = 128
    num_classes = 10

    arcface = ArcFace(
        in_features=embedding_dim, out_features=num_classes, s=64.0, m=0.50
    )

    # Create dummy embeddings and labels
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    output = arcface(embeddings, labels)

    # Check output shape
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"

    # Check that output is scaled by s
    # The output should be in a reasonable range (scaled by s)
    assert torch.all(output.abs() < 1000), "Output values seem too large"

    # Check that it's a proper PyTorch module
    assert isinstance(arcface, torch.nn.Module)
    assert hasattr(arcface, "weight")
    assert arcface.weight.shape == (num_classes, embedding_dim)


def test_arcface_margin_application():
    """Test that margin is correctly applied to target class"""
    batch_size = 2
    embedding_dim = 64
    num_classes = 5
    s = 64.0
    m = 0.50

    arcface = ArcFace(in_features=embedding_dim, out_features=num_classes, s=s, m=m)

    # Create normalized embeddings
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels = torch.tensor([0, 1])

    # Compute cosine similarity manually
    normalized_weight = F.normalize(arcface.weight, p=2, dim=1)
    cosine_sim = F.linear(embeddings, normalized_weight)

    # Forward pass
    output = arcface(embeddings, labels)

    # For target classes, the output should be scaled cosine with margin applied
    # The target class logit should be different from the cosine similarity
    for i, label in enumerate(labels):
        target_cosine = cosine_sim[i, label].item()
        target_output = output[i, label].item()

        # The output should be scaled, so it should be different from cosine
        assert (
            abs(target_output - s * target_cosine) > 1e-3
        ), f"Target class output should have margin applied, but got {target_output} vs {s * target_cosine}"

        # For non-target classes, output should equal scaled cosine
        for j in range(num_classes):
            if j != label:
                non_target_cosine = cosine_sim[i, j].item()
                non_target_output = output[i, j].item()
                assert (
                    abs(non_target_output - s * non_target_cosine) < 1e-5
                ), f"Non-target class output should equal scaled cosine, but got {non_target_output} vs {s * non_target_cosine}"


def test_arcface_scale_factor():
    """Test that scale factor s is correctly applied"""
    batch_size = 2
    embedding_dim = 32
    num_classes = 3

    # Test with different scale factors
    for s in [32.0, 64.0, 128.0]:
        arcface = ArcFace(
            in_features=embedding_dim, out_features=num_classes, s=s, m=0.50
        )
        embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        labels = torch.randint(0, num_classes, (batch_size,))

        output = arcface(embeddings, labels)

        # Compute cosine similarity
        normalized_weight = F.normalize(arcface.weight, p=2, dim=1)
        cosine_sim = F.linear(embeddings, normalized_weight)

        # For non-target classes, output should be exactly s * cosine
        for i, label in enumerate(labels):
            for j in range(num_classes):
                if j != label:
                    expected = s * cosine_sim[i, j]
                    actual = output[i, j]
                    assert torch.allclose(
                        actual, expected, atol=1e-5
                    ), f"Scale factor not applied correctly: expected {expected}, got {actual}"


def test_arcface_gradient_flow():
    """Test that gradients flow correctly through ArcFace"""
    batch_size = 2
    embedding_dim = 64
    num_classes = 5

    arcface = ArcFace(in_features=embedding_dim, out_features=num_classes)
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    output = arcface(embeddings, labels)

    # Apply softmax and compute loss
    loss = F.cross_entropy(output, labels)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert embeddings.grad is not None, "Gradients should flow to embeddings"
    assert arcface.weight.grad is not None, "Gradients should flow to weight"

    # Check that gradients are not all zeros
    assert not torch.allclose(
        embeddings.grad, torch.zeros_like(embeddings.grad)
    ), "Embedding gradients should not be all zeros"
    assert not torch.allclose(
        arcface.weight.grad, torch.zeros_like(arcface.weight.grad)
    ), "Weight gradients should not be all zeros"


def test_arcface_numerical_stability():
    """Test numerical stability with edge cases"""
    batch_size = 4
    embedding_dim = 128
    num_classes = 10

    arcface = ArcFace(
        in_features=embedding_dim, out_features=num_classes, s=64.0, m=0.50
    )

    # Test with normalized embeddings (common use case)
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    output = arcface(embeddings, labels)

    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"

    # Test with very small embeddings
    small_embeddings = torch.randn(batch_size, embedding_dim) * 1e-6
    output_small = arcface(small_embeddings, labels)
    assert not torch.isnan(
        output_small
    ).any(), "Output should handle small embeddings without NaN"

    # Test with very large embeddings
    large_embeddings = torch.randn(batch_size, embedding_dim) * 1e6
    output_large = arcface(large_embeddings, labels)
    assert not torch.isnan(
        output_large
    ).any(), "Output should handle large embeddings without NaN"


def test_arcface_different_margins():
    """Test ArcFace with different margin values"""
    batch_size = 2
    embedding_dim = 64
    num_classes = 5

    for m in [0.1, 0.3, 0.5, 0.7]:
        arcface = ArcFace(
            in_features=embedding_dim, out_features=num_classes, s=64.0, m=m
        )
        embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        labels = torch.randint(0, num_classes, (batch_size,))

        output = arcface(embeddings, labels)

        # Check output shape
        assert output.shape == (batch_size, num_classes)

        # Check for NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
