# Datasets

EasyDL provides flexible dataset classes for various data loading patterns.

## Dataset Classes

### GenericLambdaDataset

The base dataset class that uses lambda functions for data loading:

```python
from easydl.data import GenericLambdaDataset

# Create dataset with lambda functions
dataset = GenericLambdaDataset(
    lambda_dict={
        'x': lambda i: load_image(image_paths[i]),
        'y': lambda i: labels[i]
    },
    length=len(image_paths)
)

# Access data
sample = dataset[0]  # {'x': image, 'y': label}
```

### Factory Methods

Create datasets from common data structures:

```python
# From a list
dataset = GenericLambdaDataset.from_list(
    data_list=['a', 'b', 'c'],
    key='x'
)

# From x and y lists
dataset = GenericLambdaDataset.from_xy_lists(
    x_list=[img1, img2, img3],
    y_list=[0, 1, 0]
)

# From DataFrame
dataset = GenericLambdaDataset.from_dataframe(
    df=my_dataframe,
    transforms={'x': preprocess_fn},
    auto_tensorize=True
)
```

### GenericPytorchDataset

DataFrame-based dataset with automatic tensor conversion:

```python
from easydl.data import GenericPytorchDataset
import pandas as pd

df = pd.DataFrame({
    'x': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
    'y': [0, 1, 0]
})

dataset = GenericPytorchDataset(
    df=df,
    transforms={
        'x': lambda path: preprocess(load_image(path))
    }
)

# Returns torch.Tensor automatically
sample = dataset[0]  # {'x': tensor, 'y': tensor}
```

### GenericXYLambdaAutoLabelEncoderDataset

Automatically encodes string labels to integers:

```python
from easydl.data import GenericXYLambdaAutoLabelEncoderDataset

# Create from DataFrame with string labels
dataset = GenericXYLambdaAutoLabelEncoderDataset.from_df(
    df=pd.DataFrame({
        'x': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        'y': ['cat', 'dog', 'cat']
    })
)

# Labels are automatically encoded: cat=0, dog=1
print(dataset.get_number_of_classes())  # 2
print(dataset.get_y_list_with_encoded_labels())  # [0, 1, 0]
```

## Data Loading Patterns

### Basic Image Dataset

```python
from easydl.data import GenericPytorchDataset
from easydl.image import smart_read_image, COMMON_IMAGE_PREPROCESSING_FOR_TRAINING

def load_and_preprocess(path):
    image = smart_read_image(path)
    return COMMON_IMAGE_PREPROCESSING_FOR_TRAINING(image)

dataset = GenericPytorchDataset(
    df=train_df,
    transforms={'x': load_and_preprocess}
)
```

### Lazy Loading with Lambda

For large datasets, use lambda functions to load on-demand:

```python
dataset = GenericLambdaDataset(
    lambda_dict={
        'x': lambda i: load_and_preprocess(train_df.iloc[i]['x']),
        'y': lambda i: train_df.iloc[i]['y']
    },
    length=len(train_df)
)
```

### Extending Datasets

Add transformations to existing datasets:

```python
dataset = GenericLambdaDataset.from_xy_lists(x_list, y_list)

# Add preprocessing
dataset.extend_lambda_dict({
    'x': lambda img: preprocess(img)
})
```

## Using with DataLoader

```python
from torch.utils.data import DataLoader

dataset = GenericPytorchDataset(df, transforms)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for batch in dataloader:
    x = batch['x']  # [32, 3, 224, 224]
    y = batch['y']  # [32]
```

## Public Datasets

### CUB-200-2011

```python
from easydl.public_dataset.cub import CUBDataset

# Download and load CUB dataset
cub = CUBDataset(root="./data")
train_df = cub.get_train_df()
test_df = cub.get_test_df()
```

## Custom Dataset Example

Create a custom dataset for your use case:

```python
class MyImageDataset(GenericLambdaDataset):
    def __init__(self, image_dir, labels_file):
        # Load labels
        with open(labels_file) as f:
            self.labels = json.load(f)

        self.image_paths = list(self.labels.keys())

        super().__init__(
            lambda_dict={
                'x': lambda i: self._load_image(i),
                'y': lambda i: self.labels[self.image_paths[i]]
            },
            length=len(self.image_paths)
        )

    def _load_image(self, index):
        path = os.path.join(self.image_dir, self.image_paths[index])
        image = smart_read_image(path)
        return COMMON_IMAGE_PREPROCESSING_FOR_TRAINING(image)
```
