# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `dev-setup/` directory with development environment scripts:
  - `setup-dev.sh` - Main setup script for creating venv and installing dependencies
  - `install-hooks.sh` - Pre-commit hooks installation
  - `run-tests.sh` - Test runner with tier selection options
- `exp-ws/` experiment workspace for development and debugging:
  - Template experiment file
  - Dedicated `.gitignore` for experiment outputs
- Tiered test structure with four levels:
  - `tier1_unit/` - Fast unit tests (<1s)
  - `tier2_component/` - Component tests (<30s)
  - `tier3_integration/` - Integration tests with real data (<5min)
  - `tier4_e2e/` - End-to-end pipeline tests (>5min)
- New test files:
  - `tests/tier1_unit/test_image.py` - Image loading and preprocessing tests
  - `tests/tier1_unit/test_loss.py` - Loss function tests (ArcFace)
  - `tests/tier1_unit/test_infer.py` - Inference utility tests
  - `tests/tier1_unit/test_evaluation.py` - Evaluation metric tests
  - `tests/tier2_component/test_models.py` - Model component tests
  - `tests/tier2_component/test_regression.py` - Embedding regression tests
  - `tests/tier3_integration/test_evaluation_real.py` - Real data evaluation tests
  - `tests/tier4_e2e/test_training_pipeline.py` - Full training pipeline tests
- Pytest markers for test categorization: `@pytest.mark.unit`, `@pytest.mark.component`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`, `@pytest.mark.gpu`
- `pytest.ini` configuration file with marker definitions
- Shared test fixtures in `tests/conftest.py`
- Golden data regression testing with pairwise distance validation
- Development documentation:
  - `refine.md` - Codebase analysis and recommendations
  - `dev-design-v1.0.md` - Development design document
  - `test-design-v1.0.md` - Test strategy document
  - `development.md` - Developer/contributor guide
- Example scripts in `examples/`:
  - `01_quick_start.py` - Basic inference with metric learning models
  - `02_extract_embeddings.py` - Extract embeddings from images
  - `03_find_similar_images.py` - Find similar images using embeddings
  - `04_train_metric_model.py` - Train a metric learning model
  - `README.md` - Examples documentation

### Fixed
- Broken import in `tests/dml/test_evaluation.py`: changed `calculate_pr_auc_for_matrices` to `calculate_precision_recall_auc_for_pairwise_score_matrix`
- Typo in filename: renamed `easydl/dml/inferface.py` to `easydl/dml/interface.py`
- Updated imports in `easydl/common_infer.py` and `easydl/dml/hf_models.py` to use corrected module name

### Changed
- Applied `black` code formatting to all Python files (44 files reformatted)
- Applied `isort` import sorting to all Python files
- Switched to `uv` for virtual environment management
- Virtual environments now stored in `venvs/` directory (instead of `.venv`)
- Updated `pyproject.toml` with explicit package discovery to exclude non-package directories
