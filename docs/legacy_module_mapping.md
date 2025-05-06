# Legacy Module Usage Mapping

This document identifies all locations in the active codebase (excluding `historic/` directory) that still import or use code from the retired modules.

## Files with Legacy Imports

### Scripts

1. **`scripts/final_verify_event_driven.py`**
   - Imports from 4 retired modules:
     ```python
     from positronic_brain.kv_mirror import KVMirror
     from positronic_brain.culler import select_tokens_for_cull
     from positronic_brain.brightness_engine import update_brightness_scores
     from positronic_brain.context_maintenance import ContextMaintenance
     ```
   - **Category**: Substantial Logic (script using multiple core components)

2. **`scripts/inspect_capture.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load
     ```
   - **Category**: Tiny Helper (utility import for file loading)

### Test Files

3. **`tests/positronic_brain/conftest.py`**
   - Imports:
     ```python
     from positronic_brain.kv_mirror import KVMirror
     ```
   - **Category**: Test-Only Helper (fixture setup)

4. **`tests/positronic_brain/test_loop_step1.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_save
     from positronic_brain.sampler import select_next_token
     from positronic_brain.sampler_types import SamplerState
     ```
   - **Category**: Test-Only Helper (test functions)

5. **`tests/positronic_brain/test_loop_step2a_process_attention.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

6. **`tests/positronic_brain/test_loop_step2b_calc_brightness.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

7. **`tests/positronic_brain/test_loop_step3a_cull_count.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

8. **`tests/positronic_brain/test_loop_step3b_select_cull.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load, safe_save
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

9. **`tests/positronic_brain/test_loop_step3c_prepare_mlm.py`**
   - Imports:
     ```python
     from positronic_brain.serialization_utils import safe_load
     ```
   - **Category**: Test-Only Helper (file I/O utilities)

10. **`tests/positronic_brain/test_loop_step3d_run_mlm.py`**
    - Imports:
      ```python
      from positronic_brain.serialization_utils import safe_load, safe_save
      ```
    - **Category**: Test-Only Helper (file I/O utilities)

11. **`tests/positronic_brain/test_loop_step3e_consolidate_actions.py`**
    - Imports:
      ```python
      from positronic_brain.serialization_utils import safe_load, safe_save
      ```
    - **Category**: Test-Only Helper (file I/O utilities)

## Summary by Module

### 1. serialization_utils
- **Files**: 8 test files + 1 script
- **Functions**: `safe_load`, `safe_save`
- **Category**: Tiny Helper (simple file I/O utilities)
- **Usage Pattern**: Used primarily for test data storage and retrieval

### 2. kv_mirror
- **Files**: 1 script + conftest.py
- **Classes**: `KVMirror`
- **Category**: Substantial Logic (core state management)
- **Usage Pattern**: Used for token tracking and context management

### 3. sampler
- **Files**: 1 test file
- **Functions**: `select_next_token` 
- **Category**: Substantial Logic (token generation logic)
- **Usage Pattern**: Used for token selection during generation step

### 4. sampler_types
- **Files**: 1 test file
- **Classes**: `SamplerState`
- **Category**: Tiny Helper (data structure)
- **Usage Pattern**: Used to configure token sampling behavior

### 5. brightness_engine
- **Files**: 1 script
- **Functions**: `update_brightness_scores`
- **Category**: Substantial Logic (attention/brightness calculations)
- **Usage Pattern**: Used to update token importance scores

### 6. culler
- **Files**: 1 script
- **Functions**: `select_tokens_for_cull`
- **Category**: Substantial Logic (token removal logic)
- **Usage Pattern**: Used to identify tokens for removal

### 7. context_maintenance
- **Files**: 1 script
- **Classes**: `ContextMaintenance`
- **Category**: Substantial Logic (context management)
- **Usage Pattern**: Used for managing token lifetimes and relationships

## Key Observations

1. Most legacy imports in the active codebase are concentrated in test files (9 files).
2. `serialization_utils.safe_load/safe_save` is the most frequently imported functionality (9 files).
3. The script `final_verify_event_driven.py` relies heavily on legacy modules (4 different imports).
4. No direct imports found in the core library code itself, only in tests and scripts.

## Prioritization Guidance

Based on the analysis, the following migration priorities are recommended:

1. **High Priority**: Implement a replacement for `serialization_utils` functions (affects 9 files)
2. **Medium Priority**: Replace or reimplement the `KVMirror` class (affects 2 files)
3. **Lower Priority**: Address other dependencies in `test_loop_step1.py` and the verification script
