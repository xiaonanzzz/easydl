# Development Guide

This guide covers how to set up your development environment and contribute to EasyDL.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- [uv](https://github.com/astral-sh/uv) (will be installed automatically)
- (Optional) CUDA-capable GPU for training tests

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/easydl.git
cd easydl

# Run the setup script (installs uv if needed)
./dev-setup/setup-dev.sh

# Activate the virtual environment
source dev-setup/activate.sh
```

### Manual Setup (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
mkdir -p venvs
uv venv venvs/dev --python 3.9
source venvs/dev/bin/activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

## Project Structure

```
easydl/
├── easydl/                 # Main package
│   ├── __init__.py
│   ├── image.py            # Image loading utilities
│   ├── data.py             # Dataset utilities
│   ├── utils.py            # General utilities
│   ├── config.py           # Configuration classes
│   ├── common_trainer.py   # Training loops
│   ├── common_infer.py     # Inference utilities
│   ├── visualization.py    # Visualization tools
│   ├── clf/                # Classification models
│   ├── dml/                # Deep metric learning
│   ├── clustering/         # Clustering algorithms
│   └── reid/               # Re-identification
├── tests/                  # Test suite
│   ├── tier1_unit/         # Fast unit tests
│   ├── tier2_component/    # Component tests
│   ├── tier3_integration/  # Integration tests
│   └── tier4_e2e/          # End-to-end tests
├── dev-setup/              # Development scripts
├── exp-ws/                 # Experiment workspace
└── examples/               # Example notebooks
```

## Development Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Changes

Follow these guidelines:
- Keep changes focused and atomic
- Follow existing code patterns
- Add type hints for public APIs
- Write tests for new functionality

### 3. Run Tests

```bash
# Quick tests (recommended during development)
./dev-setup/run-tests.sh quick

# Full test suite
./dev-setup/run-tests.sh full

# With coverage
./dev-setup/run-tests.sh coverage
```

### 4. Format Code

```bash
# Auto-format
black easydl/ tests/
isort easydl/ tests/

# Check formatting (CI mode)
black --check easydl/
isort --check-only easydl/
```

### 5. Type Check

```bash
mypy easydl/
```

### 6. Commit Changes

```bash
# Stage changes
git add -A

# Commit (pre-commit hooks will run automatically)
git commit -m "feat: add new feature description"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 7. Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub.

## Testing

### Test Tiers

| Tier | Location | Speed | Purpose |
|------|----------|-------|---------|
| 1 | `tier1_unit/` | <1s each | Unit tests, mocked dependencies |
| 2 | `tier2_component/` | <30s each | Component integration |
| 3 | `tier3_integration/` | <5min each | Real data, real models |
| 4 | `tier4_e2e/` | >5min | Full pipeline tests |

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/tier1_unit/test_image.py -v

# Run a specific test function
pytest tests/tier1_unit/test_image.py::TestSmartReadImage::test_read_pil_image -v

# Run tests matching a pattern
pytest -k "test_cosine" -v

# Run with markers
pytest -m "unit" -v          # Only unit tests
pytest -m "not slow" -v      # Skip slow tests
pytest -m "gpu" -v           # Only GPU tests
```

### Writing Tests

```python
# tests/tier1_unit/test_example.py
import pytest
import torch

from easydl.dml.pytorch_models import Resnet18MetricModel


@pytest.mark.unit
class TestExample:
    """Unit tests for example functionality."""

    def test_basic_functionality(self):
        """Test basic case."""
        model = Resnet18MetricModel(embedding_dim=128)
        assert model.embedding.out_features == 128

    @pytest.mark.gpu
    def test_gpu_functionality(self, device):
        """Test GPU case (uses device fixture)."""
        if device.type != "cuda":
            pytest.skip("GPU not available")
        # GPU-specific test code
```

### Regression Tests

Golden data is stored in `tests/tier2_component/golden_data/`. To regenerate:

```bash
python -c "from tests.tier2_component.test_regression import generate_golden_data; generate_golden_data()"
```

## Code Style

### Formatting

- **Line length**: 88 characters (black default)
- **Imports**: Sorted with isort (black-compatible profile)
- **Quotes**: Double quotes preferred

### Type Hints

```python
from typing import Optional, List, Dict, Union

def process_images(
    images: List[str],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Process images and return embeddings.
    
    Args:
        images: List of image paths or URLs
        batch_size: Batch size for processing
        device: Device to use (default: auto-detect)
    
    Returns:
        Dictionary with 'embeddings' tensor
    """
    ...
```

### Logging

Use `smart_print()` instead of `print()`:

```python
from easydl.utils import smart_print

smart_print("Loading model...")  # Respects global print settings
```

### Exception Handling

Use explicit exceptions instead of assertions:

```python
# Good
if len(embeddings) == 0:
    raise ValueError("Embeddings cannot be empty")

# Avoid (can be disabled with -O flag)
assert len(embeddings) > 0
```

## Adding New Features

### 1. New Model

```python
# easydl/dml/pytorch_models.py

class NewMetricModel(nn.Module):
    """New metric learning model.
    
    Args:
        embedding_dim: Output embedding dimension
        pretrained: Use pretrained weights
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        # Implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns L2-normalized embeddings
        ...
```

### 2. New Loss Function

```python
# easydl/dml/loss.py

class NewLoss(nn.Module):
    """New loss function for metric learning."""
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 3. New Evaluation Metric

```python
# easydl/dml/evaluation.py

def calculate_new_metric(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Calculate new evaluation metric.
    
    Args:
        embeddings: N x D embedding matrix
        labels: N labels
    
    Returns:
        Dictionary with metric values
    """
    ...
```

## Debugging

### Using the Experiment Workspace

```bash
# Create a debug script
cp exp-ws/template.py exp-ws/debug_issue.py

# Edit and run
python exp-ws/debug_issue.py --debug
```

### Common Issues

**Import errors after changes:**
```bash
uv pip install -e ".[dev]"  # Reinstall in dev mode
```

**Test fixtures not found:**
```bash
# Make sure conftest.py is in tests/
pytest tests/ --collect-only  # Check test collection
```

**GPU memory issues:**
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller batch sizes in tests
@pytest.fixture
def small_batch():
    return torch.randn(2, 3, 224, 224)  # Small batch for testing
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create PR to main branch
4. After merge, create a git tag:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

## Resources

- [CLAUDE.md](CLAUDE.md) - AI assistant guidelines
- [dev-design-v1.0.md](dev-design-v1.0.md) - Development design document
- [test-design-v1.0.md](test-design-v1.0.md) - Test strategy document
- [CHANGELOG.md](CHANGELOG.md) - Change history
