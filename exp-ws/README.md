# Experiment Workspace

This directory is for running experiments during development and debugging.

## Purpose

Use this workspace to:
- Test new features before integrating them
- Debug training pipelines
- Run quick experiments with real data
- Prototype new models or evaluation methods

## Usage

### Quick Experiment

```python
# exp-ws/my_experiment.py
import sys
sys.path.insert(0, '..')  # Add parent to path for local easydl

from easydl.dml.pytorch_models import Resnet18MetricModel
from easydl.dml.trainer import DeepMetricLearningImageTrainverV971

# Your experiment code here...
```

### Running Experiments

```bash
# From project root
cd exp-ws
python my_experiment.py

# Or from project root
python exp-ws/my_experiment.py
```

## Directory Structure

```
exp-ws/
├── README.md           # This file
├── .gitignore          # Ignores experiment outputs
├── template.py         # Template for new experiments
└── your_experiments/   # Create subdirs for complex experiments
```

## Guidelines

1. **Keep experiments temporary** - This is a workspace, not permanent storage
2. **Don't commit large files** - Model checkpoints, datasets, etc. are gitignored
3. **Use relative imports** - Import from `easydl` package for testing local changes
4. **Clean up** - Remove old experiments periodically

## Gitignore

The following are automatically ignored:
- `*.pth` - Model checkpoints
- `*.pt` - PyTorch files
- `*.pkl` - Pickle files
- `*.npy` - NumPy arrays
- `data/` - Local datasets
- `outputs/` - Experiment outputs
- `logs/` - Training logs
- `wandb/` - W&B logs

## Example Workflow

```bash
# 1. Make changes to easydl source
vim easydl/dml/trainer.py

# 2. Test in experiment workspace
cd exp-ws
python test_trainer_changes.py

# 3. If it works, write proper tests
vim tests/tier2_component/test_new_feature.py

# 4. Clean up experiment
rm exp-ws/test_trainer_changes.py
```
