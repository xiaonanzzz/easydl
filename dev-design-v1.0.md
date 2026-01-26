# EasyDL v1.0 Development Design Document

**Version**: 1.0
**Branch**: dev-1.0
**Status**: Draft
**Last Updated**: 2026-01-26

---

## 1. Overview

### 1.1 Purpose
This document outlines the development plan for EasyDL v1.0, focusing on code quality improvements, bug fixes, and architectural refinements to establish a stable foundation for future development.

### 1.2 Goals
- Establish consistent code quality standards across the codebase
- Fix critical bugs and broken functionality
- Improve type safety and documentation
- Enhance performance for production use cases
- Increase test coverage to ensure reliability

### 1.3 Non-Goals
- Adding new features (deferred to v1.1+)
- Major API changes
- Breaking backward compatibility

---

## 2. Current State Analysis

### 2.1 Codebase Metrics
| Metric | Value |
|--------|-------|
| Total Lines of Code | ~4,004 |
| Total Test Lines | ~1,006 |
| Python Modules | 27 |
| Test Files | 7 |
| Functions | 167 |
| Classes | 51 |

### 2.2 Quality Issues Identified
| Category | Count | Severity |
|----------|-------|----------|
| Critical Bugs | 2 | High |
| Formatting Issues | 26 files | Medium |
| Type Hint Errors | 10 | Medium |
| Missing Documentation | 7 modules | Low |
| Test Coverage Gaps | 4 modules | Medium |

---

## 3. Design Decisions

### 3.1 Code Style Standards

**Decision**: Adopt strict formatting with `black` and `isort`.

**Rationale**:
- Eliminates style debates in code reviews
- Ensures consistent readability
- Enables automated CI checks

**Configuration**:
```toml
# pyproject.toml additions
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']

[tool.isort]
profile = "black"
line_length = 88
```

### 3.2 Type Hints Strategy

**Decision**: Require type hints for all public APIs; use `Optional[]` correctly.

**Rationale**:
- Enables static analysis with mypy
- Improves IDE support and autocompletion
- Self-documenting code

**Pattern**:
```python
# Before
def func(param: str = None) -> str:
    ...

# After
from typing import Optional

def func(param: Optional[str] = None) -> str:
    ...
```

### 3.3 Logging Architecture

**Decision**: Standardize on `smart_print()` for all logging output.

**Rationale**:
- Configurable output destination (console, file, both)
- Consistent logging interface
- Production-ready logging control

**Migration**:
```python
# Before
print(f"Loading model from {path}")

# After
from easydl.utils import smart_print
smart_print(f"Loading model from {path}")
```

### 3.4 Exception Handling

**Decision**: Replace assertions with explicit exceptions in production code.

**Rationale**:
- Assertions can be disabled with `-O` flag
- Explicit exceptions provide better error messages
- Proper exception types enable targeted error handling

**Pattern**:
```python
# Before
assert len(embeddings) > 0

# After
if len(embeddings) == 0:
    raise ValueError("Embeddings cannot be empty")
```

### 3.5 Interface Naming

**Decision**: Rename `inferface.py` to `interface.py`.

**Rationale**:
- Fix typo for code clarity
- Prevent confusion and import errors
- Professional code quality

---

## 4. Implementation Phases

### Phase 1: Critical Fixes (Week 1)

#### 4.1.1 Fix Broken Test Import
- **File**: `tests/dml/test_evaluation.py`
- **Change**: Update import from `calculate_pr_auc_for_matrices` to `calculate_precision_recall_auc_for_pairwise_score_matrix`
- **Verification**: `pytest tests/dml/test_evaluation.py -v`

#### 4.1.2 Rename Interface File
- **From**: `easydl/dml/inferface.py`
- **To**: `easydl/dml/interface.py`
- **Update**: All imports referencing this module
- **Verification**: `grep -r "inferface" easydl/`

#### 4.1.3 Apply Code Formatting
```bash
black easydl/ tests/
isort easydl/ tests/
```
- **Verification**: `black --check easydl/ && isort --check-only easydl/`

### Phase 2: Type Safety (Week 2)

#### 4.2.1 Type Hint Fixes
| File | Change |
|------|--------|
| `dml/simulation.py:4` | `random_seed: int = None` → `Optional[int] = None` |
| `dml/evaluation.py:294` | `evaluation_report_dir: str = None` → `Optional[str] = None` |
| `image.py:135` | Fix `base64.binascii` attribute access |
| `visualization.py:86,113` | Add missing type annotations |
| `data.py:69` | `callable` → `typing.Callable` |

#### 4.2.2 Remove Wildcard Import
- **File**: `utils.py:6`
- **Change**: Replace `from easydl.config import *` with explicit imports

### Phase 3: Code Quality (Week 3)

#### 4.3.1 Logging Standardization
Replace `print()` with `smart_print()` in:
- `dml/pytorch_models.py:171-174, 283-286`
- `dml/evaluation.py`
- `visualization.py:15`
- `dml/trainer.py`

#### 4.3.2 Exception Handling Refactor
- **File**: `dml/pytorch_models.py:168-174`
- **Issue**: Redundant exception handling that retries identical operation
- **Solution**: Implement proper fallback logic or remove redundant try-except

#### 4.3.3 Replace Assertions
Convert assertions to explicit exceptions in:
- `common_infer.py:11`
- `dml/evaluation.py:97`
- `dml/pytorch_models.py:147`

### Phase 4: Documentation (Week 4)

#### 4.4.1 Add Module Docstrings
Add docstrings to:
- `dml/interface.py` (formerly inferface.py)
- `clf/pytorch_models.py`
- `common_infer.py`
- `numpyext.py`
- `clustering/__init__.py`
- `clf/image_net.py`
- `reid/clip_reid_config.py`

#### 4.4.2 Fix README
- Update optional dependency references
- Replace `[core]`, `[infer]`, `[train]` with actual extras: `[research]`, `[dev]`, `[all]`

### Phase 5: Testing (Week 5)

#### 4.5.1 Add Missing Tests
| Module | Priority |
|--------|----------|
| `utils.py` | High |
| `config.py` | High |
| `clustering/` | Medium |
| `common_trainer.py` | Medium |

#### 4.5.2 Test Coverage Target
- Current: ~60%
- Target: 80%

### Phase 6: Performance (Week 6)

#### 4.6.1 GPU Memory Optimization
- **File**: `dml/evaluation.py:17-49`
- **Solution**: Implement batched GPU operations for large embedding matrices
```python
def compute_similarity_batched(embeddings, batch_size=1000):
    """Compute similarity matrix in batches to avoid OOM."""
    n = len(embeddings)
    similarity = torch.zeros(n, n)
    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            batch_i = embeddings[i:i+batch_size]
            batch_j = embeddings[j:j+batch_size]
            similarity[i:i+batch_size, j:j+batch_size] = batch_i @ batch_j.T
    return similarity
```

#### 4.6.2 Async Image Loading (Optional)
- **File**: `image.py`
- **Solution**: Add optional async support for batch image loading
- **Deferred**: May move to v1.1 based on timeline

---

## 5. File Changes Summary

### 5.1 Files to Modify
| File | Changes |
|------|---------|
| `tests/dml/test_evaluation.py` | Fix import |
| `easydl/dml/simulation.py` | Type hints |
| `easydl/dml/evaluation.py` | Type hints, assertions, logging |
| `easydl/dml/pytorch_models.py` | Exception handling, logging, assertions |
| `easydl/image.py` | Type hints |
| `easydl/visualization.py` | Type hints, logging |
| `easydl/data.py` | Type hints |
| `easydl/utils.py` | Remove wildcard import |
| `easydl/common_infer.py` | Assertions, docstring |
| `README.md` | Fix dependency references |

### 5.2 Files to Rename
| From | To |
|------|-----|
| `easydl/dml/inferface.py` | `easydl/dml/interface.py` |

### 5.3 Files to Create
| File | Purpose |
|------|---------|
| `tests/unit_tests/test_utils.py` | Unit tests for utils module |
| `tests/unit_tests/test_config.py` | Unit tests for config module |
| `tests/unit_tests/test_clustering.py` | Unit tests for clustering module |

---

## 6. CI/CD Integration

### 6.1 Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### 6.2 GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: black --check easydl/
      - run: isort --check-only easydl/
      - run: mypy easydl/
      - run: pytest tests/ -v
```

---

## 7. Success Criteria

### 7.1 Phase Completion Criteria
| Phase | Criteria |
|-------|----------|
| Phase 1 | All tests pass, no formatting errors |
| Phase 2 | `mypy easydl/` reports 0 errors |
| Phase 3 | No `print()` calls except in demo code |
| Phase 4 | All public modules have docstrings |
| Phase 5 | Test coverage >= 80% |
| Phase 6 | Embedding evaluation handles 100k+ samples |

### 7.2 Release Criteria
- [ ] All CI checks pass
- [ ] No critical or high severity issues
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped to 1.0.0

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking imports after rename | High | Update all references, add deprecation warning |
| Performance regression | Medium | Benchmark before/after changes |
| Test failures during refactor | Medium | Incremental commits, CI on each PR |

---

## 9. Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Critical Fixes | Broken tests fixed, formatting applied |
| 2 | Type Safety | All type hints correct, mypy passes |
| 3 | Code Quality | Logging standardized, exceptions refactored |
| 4 | Documentation | All docstrings added, README updated |
| 5 | Testing | New tests written, coverage increased |
| 6 | Performance | GPU optimization implemented |

---

## 10. Appendix

### 10.1 Quick Commands Reference
```bash
# Development setup
pip install -e ".[dev]"

# Formatting
black easydl/ tests/
isort easydl/ tests/

# Type checking
mypy easydl/

# Testing
pytest tests/ -v
pytest tests/ --cov=easydl --cov-report=html

# Verify all checks
black --check easydl/ && isort --check-only easydl/ && mypy easydl/ && pytest tests/
```

### 10.2 Related Documents
- `refine.md` - Initial codebase analysis
- `CLAUDE.md` - Development guidelines
- `README.md` - Project documentation
