# Legacy Surface Map

This document maps all locations in the active codebase where retired modules are still being imported and used. This analysis excludes code in the `historic/` directory.

## Overview

The legacy surface analysis identified **11 active files** that contain imports from retired modules:
- 9 test files
- 2 script files

## Files with Legacy Imports

### Scripts

1. **./scripts/final_verify_event_driven.py**
   - Imports:
     ```python
     from positronic_brain.kv_mirror import KVMirror
     from positronic_brain.culler import select_tokens_for_cull
     from positronic_brain.brightness_engine import update_brightness_scores
     from positronic_brain.context_maintenance import ContextMaintenance
     ```
   - **Category**: Substantial Logic (script using multiple core components)

2. **./scripts/inspect_capture.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load
     ```
   - **Category**: Tiny Helper (utility import for file loading)

### Test Files

3. **./tests/positronic_brain/conftest.py**
   - Imports:
     ```python
     from positronic_brain.kv_mirror import KVMirror
     ```
   - **Category**: Test-Only Helper (fixture setup)

4. **./tests/positronic_brain/test_loop_step1.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_save
     from positronic_brain.sampler import select_next_token
     from positronic_brain.sampler_types import SamplerState
     ```
   - **Category**: Test-Only Helper

5. **./tests/positronic_brain/test_loop_step2a_process_attention.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

6. **./tests/positronic_brain/test_loop_step2b_calc_brightness.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

7. **./tests/positronic_brain/test_loop_step3a_cull_count.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

8. **./tests/positronic_brain/test_loop_step3b_select_cull.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

9. **./tests/positronic_brain/test_loop_step3c_prepare_mlm.py**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

10. **./tests/positronic_brain/test_loop_step3d_run_mlm.py**
    - Imports:
      ```python
      from positronic_brain.serialization_utils import safe_load, safe_save
      ```
    - **Category**: Test-Only Helper (file I/O utilities)

11. **./tests/positronic_brain/test_loop_step3e_consolidate_actions.py**
    - Imports:
      ```python
      from positronic_brain.serialization_utils import safe_load, safe_save
      ```
    - **Category**: Test-Only Helper (file I/O utilities)

## Summary by Module

### 1. serialization_utils

- **Import Count**: 9 files (8 test files + 1 script)
- **Functions Used**: `safe_load`, `safe_save`
- **Category**: Tiny Helper
- **Purpose**: Simple file I/O utilities
- **Files Using**:
  - `./scripts/inspect_capture.py`
  - `./tests/positronic_brain/test_loop_step1.py`
  - `./tests/positronic_brain/test_loop_step2a_process_attention.py`
  - `./tests/positronic_brain/test_loop_step2b_calc_brightness.py`
  - `./tests/positronic_brain/test_loop_step3a_cull_count.py`
  - `./tests/positronic_brain/test_loop_step3b_select_cull.py`
  - `./tests/positronic_brain/test_loop_step3c_prepare_mlm.py`
  - `./tests/positronic_brain/test_loop_step3d_run_mlm.py`
  - `./tests/positronic_brain/test_loop_step3e_consolidate_actions.py`

### 2. kv_mirror

- **Import Count**: 2 files (1 script + 1 test fixture)
- **Classes Used**: `KVMirror`
- **Category**: Substantial Logic
- **Purpose**: Core state management
- **Files Using**:
  - `./scripts/final_verify_event_driven.py`
  - `./tests/positronic_brain/conftest.py`

### 3. sampler

- **Import Count**: 1 file (test)
- **Functions Used**: `select_next_token`
- **Category**: Substantial Logic
- **Purpose**: Token generation logic
- **Files Using**:
  - `./tests/positronic_brain/test_loop_step1.py`

### 4. sampler_types

- **Import Count**: 1 file (test)
- **Classes Used**: `SamplerState`
- **Category**: Tiny Helper
- **Purpose**: Data structure
- **Files Using**:
  - `./tests/positronic_brain/test_loop_step1.py`

### 5. brightness_engine

- **Import Count**: 1 file (script)
- **Functions Used**: `update_brightness_scores`
- **Category**: Substantial Logic
- **Purpose**: Attention/brightness calculations
- **Files Using**:
  - `./scripts/final_verify_event_driven.py`

### 6. culler

- **Import Count**: 1 file (script)
- **Functions Used**: `select_tokens_for_cull`
- **Category**: Substantial Logic
- **Purpose**: Token removal logic
- **Files Using**:
  - `./scripts/final_verify_event_driven.py`

### 7. context_maintenance

- **Import Count**: 1 file (script)
- **Classes Used**: `ContextMaintenance`
- **Category**: Substantial Logic
- **Purpose**: Context management
- **Files Using**:
  - `./scripts/final_verify_event_driven.py`

## Key Observations

1. Most legacy imports in the active codebase are concentrated in test files (9 files).
2. `serialization_utils.safe_load/safe_save` is the most frequently imported functionality (9 files).
3. The script `final_verify_event_driven.py` relies heavily on legacy modules (4 different imports).
4. No direct imports found in the core library code itself, only in tests and scripts.
5. There are no dead imports - all imported components appear to be actively used.

## Prioritization Recommendation

Based on this analysis, recommended prioritization for migration:

1. **High Priority**: `serialization_utils` (most widely used, easiest to replace)
2. **Medium Priority**: `kv_mirror` (used in test fixtures)
3. **Medium Priority**: `sampler` and `sampler_types` (needed for Step 1 tests)
4. **Low Priority**: `final_verify_event_driven.py` script and its dependencies
