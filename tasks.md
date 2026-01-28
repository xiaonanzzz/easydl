# Image Pair Matching Model - Implementation Tasks

## Overview

Implement a binary classification model that determines whether two sets of dog/cat images belong to the same individual animal. The model uses transformer token embeddings as input features for a reasoning head.

### Architecture Summary

```
┌─────────────────┐     ┌─────────────────┐
│  Image Set A    │     │  Image Set B    │
│  (N images)     │     │  (M images)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│     Shared Transformer Backbone          │
│  (e.g., ViT - extract last layer tokens) │
└────────┬───────────────────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Token Pooling  │     │  Token Pooling  │
│  (Set → Vector) │     │  (Set → Vector) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    Reasoning Head     │
         │ (Binary Classifier)   │
         └───────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Same/Diff  │
              │  (0 or 1)   │
              └─────────────┘
```

---

## Task 0: Dataset Class Refactoring (Prerequisite)

**Goal**: Unify all dataset classes under `GenericLambdaDataset` as the base class while maintaining backward compatibility.

### 0.1 Refactor Class Hierarchy

**File**: `easydl/data.py`

**Current Structure**:
```
GenericPytorchDataset (standalone)
GenericLambdaDataset (standalone)
└── GenericXYLambdaAutoLabelEncoderDataset
```

**Target Structure**:
```
GenericLambdaDataset (base class)
├── from_dataframe()              # Factory method
├── from_list()                   # Factory method
├── from_xy_lists()               # Factory method
│
├── GenericPytorchDataset         # Inherits from GenericLambdaDataset
│   (backward compatible wrapper with auto-tensorization)
│
└── GenericXYLambdaAutoLabelEncoderDataset
    (unchanged, already inherits from GenericLambdaDataset)
```

### 0.2 Add Factory Methods to GenericLambdaDataset

**File**: `easydl/data.py`

```python
class GenericLambdaDataset(Dataset):
    # ... existing __init__, __len__, __getitem__ ...

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        transforms: Optional[Dict[str, Callable]] = None,
        auto_tensorize: bool = False
    ) -> "GenericLambdaDataset":
        """
        Create a GenericLambdaDataset from a DataFrame.

        Args:
            df: DataFrame containing the data
            transforms: Dict mapping column names to transform functions
            auto_tensorize: If True, automatically convert outputs to torch.Tensor

        Returns:
            GenericLambdaDataset instance
        """
        pass

    @staticmethod
    def from_list(data_list: list, key: str = "x") -> "GenericLambdaDataset":
        """
        Create dataset from a simple list.

        Args:
            data_list: List of data items
            key: Key name for the data in output dict

        Returns:
            GenericLambdaDataset instance
        """
        return GenericLambdaDataset(
            lambda_dict={key: lambda i: data_list[i]},
            length=len(data_list)
        )

    @staticmethod
    def from_xy_lists(x_list: list, y_list: list) -> "GenericLambdaDataset":
        """
        Create dataset from separate x and y lists.

        Args:
            x_list: List of input data
            y_list: List of labels (must match length of x_list)

        Returns:
            GenericLambdaDataset instance
        """
        assert len(x_list) == len(y_list), "x_list and y_list must have same length"
        return GenericLambdaDataset(
            lambda_dict={
                "x": lambda i: x_list[i],
                "y": lambda i: y_list[i]
            },
            length=len(x_list)
        )
```

### 0.3 Refactor GenericPytorchDataset to Inherit from GenericLambdaDataset

**File**: `easydl/data.py`

```python
class GenericPytorchDataset(GenericLambdaDataset):
    """
    A DataFrame-based Dataset with automatic tensor conversion.

    This class maintains full backward compatibility with existing code
    while being built on top of GenericLambdaDataset.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        transforms (dict): A dictionary where keys are column names of the DataFrame,
            and values are functions that transform the raw values in those columns
            to PyTorch tensors. If a column's key is not in the dict, the original
            value is passed as is.
    """

    def __init__(self, df: pd.DataFrame, transforms: Optional[Dict[str, Callable]] = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        self.df = df
        self.transforms = transforms if transforms is not None else {}
        self.columns = df.columns

        # Build lambda dict from DataFrame columns
        lambda_dict = {}
        for col in self.columns:
            lambda_dict[col] = self._make_column_loader(col)

        super().__init__(lambda_dict, length=len(df))

    def _make_column_loader(self, col: str) -> Callable[[int], torch.Tensor]:
        """
        Create a lambda function that loads and transforms a column value.

        Args:
            col: Column name

        Returns:
            Lambda function that takes index and returns transformed tensor
        """
        def loader(index: int) -> torch.Tensor:
            value = self.df.iloc[index][col]
            if col in self.transforms:
                try:
                    value = self.transforms[col](value)
                except Exception as e:
                    raise RuntimeError(
                        f"Error applying transform to column '{col}' at index {index}: {e}"
                    ) from e
            # Auto-convert to tensor (backward compatibility)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            return value
        return loader
```

### 0.4 Unit Tests for Refactored Dataset Classes

**File**: `tests/tier1_unit/test_data_refactor.py`

```python
def test_generic_pytorch_dataset_backward_compatibility():
    """Ensure GenericPytorchDataset works exactly as before."""
    pass

def test_generic_pytorch_dataset_inherits_from_lambda():
    """Verify inheritance relationship."""
    assert issubclass(GenericPytorchDataset, GenericLambdaDataset)

def test_from_dataframe_factory():
    """Test GenericLambdaDataset.from_dataframe() factory method."""
    pass

def test_from_list_factory():
    """Test GenericLambdaDataset.from_list() factory method."""
    pass

def test_from_xy_lists_factory():
    """Test GenericLambdaDataset.from_xy_lists() factory method."""
    pass

def test_extend_lambda_dict_works_on_pytorch_dataset():
    """Verify that GenericPytorchDataset inherits extend_lambda_dict()."""
    pass

def test_get_value_from_key_and_index_works_on_pytorch_dataset():
    """Verify that GenericPytorchDataset inherits helper methods."""
    pass
```

### 0.5 Benefits of Refactoring

| Benefit | Description |
|---------|-------------|
| **Unified Base** | All datasets share the same underlying logic |
| **Composability** | `extend_lambda_dict()` works on all dataset types |
| **Flexibility** | Factory methods for common patterns |
| **Backward Compatible** | `GenericPytorchDataset` API unchanged |
| **Extensibility** | New datasets (like `PairSamplerDataset`) can inherit common functionality |

---

## Task 1: Pair Dataset Implementation

### 1.1 Create `PairSamplerDataset` Class

**File**: `easydl/data.py`

**Purpose**: Generate pairs of image sets from a labeled dataset for training the binary classifier.

**Requirements**:
- **Inherits from `GenericLambdaDataset`** (uses refactored base class from Task 0)
- Input: DataFrame with columns `['x', 'y']` where `x` is image path and `y` is the label (same label = same animal)
- Output per sample: `{'set_a': Tensor, 'set_b': Tensor, 'label': int}` where `label=1` if same animal, `label=0` otherwise
- Support configurable number of images per set (e.g., 1-5 images per set)
- Support balanced sampling (equal positive/negative pairs)
- Handle edge cases: labels with only one image, insufficient images for a set

**Interface**:
```python
class PairSamplerDataset(GenericLambdaDataset):
    """
    Dataset that generates pairs of image sets for binary matching.

    Inherits from GenericLambdaDataset, using lambda functions to
    dynamically sample positive/negative pairs on each access.
    """

    @staticmethod
    def from_labeled_df(
        df: pd.DataFrame,
        num_pairs: int,
        images_per_set: int = 3,
        positive_ratio: float = 0.5,
        transform: Optional[Callable] = None,
        seed: int = 42
    ) -> "PairSamplerDataset":
        """
        Factory method to create PairSamplerDataset from a labeled DataFrame.

        Args:
            df: DataFrame with 'x' (image path) and 'y' (label) columns
            num_pairs: Total number of pairs in the dataset
            images_per_set: Number of images to sample for each set
            positive_ratio: Ratio of positive pairs (same animal) vs negative
            transform: Image transformation function
            seed: Random seed for reproducibility

        Returns:
            PairSamplerDataset instance
        """
        pass

    def __init__(
        self,
        lambda_dict: Dict[str, Callable[[int], Any]],
        length: int,
        label_to_indices: Dict[Any, List[int]],
        images_per_set: int,
        positive_ratio: float,
        transform: Optional[Callable],
        rng: np.random.Generator
    ):
        """
        Internal constructor. Use from_labeled_df() factory method instead.
        """
        super().__init__(lambda_dict, length)
        # Store additional attributes for pair sampling logic
        self.label_to_indices = label_to_indices
        self.images_per_set = images_per_set
        self.positive_ratio = positive_ratio
        self.transform = transform
        self.rng = rng
```

### 1.2 Create Helper Functions for Pair Sampling

**File**: `easydl/data.py`

**Functions to implement**:
```python
def sample_positive_pair(df: pd.DataFrame, images_per_set: int) -> Tuple[List[str], List[str], int]:
    """Sample two sets of images from the same label."""
    pass

def sample_negative_pair(df: pd.DataFrame, images_per_set: int) -> Tuple[List[str], List[str], int]:
    """Sample two sets of images from different labels."""
    pass

def create_pair_dataset_from_labeled_df(
    df: pd.DataFrame,
    num_pairs: int,
    images_per_set: int = 3,
    positive_ratio: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a DataFrame of image pairs for training.

    Returns DataFrame with columns:
    - 'set_a': List of image paths for set A
    - 'set_b': List of image paths for set B
    - 'label': 1 if same animal, 0 otherwise
    """
    pass
```

### 1.3 Unit Tests for Dataset

**File**: `tests/tier1_unit/test_pair_dataset.py`

- Test positive pair sampling (same label)
- Test negative pair sampling (different labels)
- Test balanced sampling ratio
- Test edge cases (single image labels, insufficient images)
- Test reproducibility with seed

---

## Task 2: Model Architecture

### 2.1 Create Token Extraction Backbone

**File**: `easydl/dml/pytorch_models.py` (or new file `easydl/pair_matching/models.py`)

**Purpose**: Extract last-layer transformer tokens from images.

**Requirements**:
- Use existing ViT models from `pytorch_models.py` or `hf_models.py`
- Extract all tokens from the last transformer layer (not just CLS token)
- Return shape: `(batch_size, num_tokens, hidden_dim)` e.g., `(B, 197, 768)` for ViT-B

**Interface**:
```python
class TokenExtractorBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_b_16",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224)
        Returns:
            tokens: (batch_size, num_tokens, hidden_dim)
        """
        pass

    def get_token_dim(self) -> int:
        """Return the hidden dimension of tokens."""
        pass

    def get_num_tokens(self) -> int:
        """Return the number of tokens per image."""
        pass
```

### 2.2 Create Set Pooling Module

**File**: `easydl/pair_matching/models.py`

**Purpose**: Aggregate multiple image tokens into a single set representation.

**Options to implement**:
1. **Mean Pooling**: Average all tokens across all images in the set
2. **Attention Pooling**: Learn weighted aggregation of tokens
3. **CLS Token Only**: Use only CLS tokens from each image, then aggregate

**Interface**:
```python
class SetPoolingModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pooling_type: str = "attention"  # "mean", "attention", "cls_mean"
    ):
        pass

    def forward(self, token_sets: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_sets: List of tensors, each (num_images, num_tokens, hidden_dim)
        Returns:
            pooled: (batch_size, output_dim)
        """
        pass
```

### 2.3 Create Reasoning Head

**File**: `easydl/pair_matching/models.py`

**Purpose**: Compare two set representations and predict binary similarity.

**Options to implement**:
1. **Concatenation + MLP**: `[set_a; set_b; |set_a - set_b|; set_a * set_b]` → MLP → sigmoid
2. **Cross-Attention**: Attention between set representations → MLP → sigmoid
3. **Siamese + Distance**: Cosine similarity or learned distance metric

**Interface**:
```python
class ReasoningHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        reasoning_type: str = "concat_mlp"  # "concat_mlp", "cross_attention", "siamese"
    ):
        pass

    def forward(self, set_a: torch.Tensor, set_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            set_a: (batch_size, embed_dim)
            set_b: (batch_size, embed_dim)
        Returns:
            logits: (batch_size, 1) - raw logits for binary classification
        """
        pass
```

### 2.4 Create Complete Model

**File**: `easydl/pair_matching/models.py`

**Interface**:
```python
class PairMatchingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_b_16",
        pooling_type: str = "attention",
        reasoning_type: str = "concat_mlp",
        embed_dim: int = 256,
        freeze_backbone: bool = False
    ):
        pass

    def forward(
        self,
        image_set_a: torch.Tensor,  # (batch, num_images, 3, H, W)
        image_set_b: torch.Tensor   # (batch, num_images, 3, H, W)
    ) -> torch.Tensor:
        """Returns logits (batch_size, 1)"""
        pass

    def predict(
        self,
        image_set_a: torch.Tensor,
        image_set_b: torch.Tensor
    ) -> torch.Tensor:
        """Returns probabilities (batch_size,)"""
        return torch.sigmoid(self.forward(image_set_a, image_set_b)).squeeze(-1)
```

### 2.5 Model Tests

**File**: `tests/tier2_component/test_pair_matching_model.py`

- Test forward pass shapes
- Test with different backbone configurations
- Test with different pooling types
- Test with different reasoning types
- Test gradient flow (unfrozen vs frozen backbone)

---

## Task 3: Training Pipeline

### 3.1 Create Training Function

**File**: `easydl/pair_matching/trainer.py`

**Interface**:
```python
def train_pair_matching_model(
    model: PairMatchingModel,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    num_pairs_per_epoch: int = 10000,
    images_per_set: int = 3,
    positive_ratio: float = 0.5,
    batch_size: int = 32,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    use_accelerator: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train the pair matching model.

    Args:
        model: PairMatchingModel instance
        train_df: DataFrame with 'x' and 'y' columns
        val_df: Optional validation DataFrame
        num_pairs_per_epoch: Number of pairs to sample per epoch
        images_per_set: Images per set in each pair
        positive_ratio: Ratio of positive pairs
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        use_accelerator: Use HuggingFace Accelerate
        save_dir: Directory to save checkpoints
        seed: Random seed

    Returns:
        Dict with training history and final metrics
    """
    pass
```

### 3.2 Create Loss Function

**File**: `easydl/pair_matching/loss.py`

```python
class PairMatchingLoss(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Binary cross-entropy loss with optional class weighting.

        Args:
            pos_weight: Weight for positive class (for imbalanced data)
        """
        pass

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass
```

### 3.3 Create Evaluation Function

**File**: `easydl/pair_matching/evaluation.py`

```python
def evaluate_pair_matching_model(
    model: PairMatchingModel,
    eval_df: pd.DataFrame,
    num_pairs: int = 1000,
    images_per_set: int = 3,
    batch_size: int = 32,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate pair matching model.

    Returns:
        Dict with metrics:
        - accuracy
        - precision
        - recall
        - f1_score
        - auc_roc
        - auc_pr
    """
    pass
```

---

## Task 4: Integration with Existing Framework

### 4.1 Add Model to DMLModelManager

**File**: `easydl/dml/pytorch_models.py`

Update `DMLModelManager.get_model()` to support pair matching models:
```python
# Add support for:
DMLModelManager.get_model("pair_matching_vit_b_16", ...)
```

### 4.2 Create High-Level Training Interface

**File**: `easydl/pair_matching/trainer.py`

```python
class PairMatchingTrainer:
    """
    High-level trainer following EasyDL's "Three lines P99" principle.

    Usage:
        trainer = PairMatchingTrainer(model_name="vit_b_16")
        trainer.train(train_df, num_epochs=50)
        results = trainer.evaluate(test_df)
    """

    def __init__(
        self,
        model_name: str = "vit_b_16",
        pooling_type: str = "attention",
        reasoning_type: str = "concat_mlp",
        embed_dim: int = 256,
        freeze_backbone: bool = False
    ):
        pass

    def train(self, train_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        pass

    def evaluate(self, test_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        pass

    def predict(self, image_set_a: List[str], image_set_b: List[str]) -> float:
        """Predict probability that two image sets are same animal."""
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
```

---

## Task 5: Documentation and Examples

### 5.1 Create Usage Example

**File**: `examples/pair_matching_example.py`

```python
"""
Example: Train a model to identify if two sets of pet photos are the same animal.
"""

from easydl.pair_matching import PairMatchingTrainer
import pandas as pd

# Load your labeled dataset
# Each image has a label - images with same label are same animal
train_df = pd.read_csv("pet_dataset.csv")  # columns: ['x', 'y']

# Initialize trainer
trainer = PairMatchingTrainer(
    model_name="vit_b_16",
    pooling_type="attention",
    reasoning_type="concat_mlp"
)

# Train
trainer.train(
    train_df=train_df,
    num_epochs=50,
    batch_size=32
)

# Evaluate
metrics = trainer.evaluate(test_df)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Inference
prob = trainer.predict(
    image_set_a=["dog1_photo1.jpg", "dog1_photo2.jpg"],
    image_set_b=["dog2_photo1.jpg", "dog2_photo2.jpg"]
)
print(f"Same animal probability: {prob:.4f}")
```

### 5.2 Add Module Docstrings

Add comprehensive docstrings to all new modules following existing patterns in the codebase.

---

## Task 6: Testing

### 6.1 Unit Tests (Tier 1)

**File**: `tests/tier1_unit/test_pair_matching/`

- `test_pair_dataset.py`: Dataset sampling logic
- `test_models.py`: Model component shapes and outputs
- `test_loss.py`: Loss function behavior

### 6.2 Component Tests (Tier 2)

**File**: `tests/tier2_component/test_pair_matching/`

- `test_model_forward.py`: Full model forward pass
- `test_backbone_tokens.py`: Token extraction verification

### 6.3 Integration Tests (Tier 3)

**File**: `tests/tier3_integration/test_pair_matching/`

- `test_training_loop.py`: Full training on synthetic data
- `test_evaluation.py`: Evaluation pipeline

---

## Implementation Order

1. **Phase 0: Dataset Refactoring** (Task 0) - *Prerequisite*
   - Add factory methods to `GenericLambdaDataset`
   - Refactor `GenericPytorchDataset` to inherit from `GenericLambdaDataset`
   - Write backward compatibility tests
   - Verify existing tests still pass

2. **Phase 1: Pair Dataset** (Task 1)
   - Implement pair sampling functions
   - Create `PairSamplerDataset` class (inheriting from `GenericLambdaDataset`)
   - Write unit tests

3. **Phase 2: Model Components** (Task 2.1 - 2.3)
   - Token extraction backbone
   - Set pooling module
   - Reasoning head

4. **Phase 3: Complete Model** (Task 2.4 - 2.5)
   - Assemble complete `PairMatchingModel`
   - Write model tests

5. **Phase 4: Training Pipeline** (Task 3)
   - Loss function
   - Training function
   - Evaluation function

6. **Phase 5: Integration** (Task 4)
   - High-level trainer interface
   - Integration with existing framework

7. **Phase 6: Documentation** (Task 5)
   - Usage example
   - Module docstrings

---

## Open Questions for Clarification

1. **Set Size Flexibility**: Should the model support variable-sized image sets, or fixed size only?

2. **Backbone Choice**: Preference for specific ViT variant (B/16, B/32, L/16)?

3. **Token Usage**:
   - Use all transformer tokens (including CLS + patch tokens)?
   - Or only CLS token from each image?

4. **Training Strategy**:
   - End-to-end training vs two-stage (frozen backbone first, then fine-tune)?
   - Hard negative mining for better discrimination?

5. **Memory Constraints**: Expected batch size and number of images per set? This affects architecture choices.

6. **Inference Optimization**: Need for ONNX export or TorchScript compilation?
