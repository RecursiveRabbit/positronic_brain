import pytest

def test_no_historic_importable():
    # Test that the old 'historic' name is not importable
    try:
        import historic  # Try to import the top-level 'historic' package
        # If the import succeeds, it means 'historic' is in PYTHONPATH, which is wrong.
        pytest.fail("The 'historic' directory should NOT be importable or on PYTHONPATH.")
    except ImportError:
        # This is the expected behavior if 'historic' is correctly isolated.
        pass
    except Exception as e:
        pytest.fail(f"Unexpected error during 'historic' import check: {e}")
    
    # Also test that the new '_historic' name is not importable
    try:
        import _historic  # Try to import the renamed directory
        # If the import succeeds, we still have the same problem
        pytest.fail("The '_historic' directory should NOT be importable or on PYTHONPATH.")
    except ImportError:
        # This is the expected behavior
        pass
    except Exception as e:
        pytest.fail(f"Unexpected error during '_historic' import check: {e}")
