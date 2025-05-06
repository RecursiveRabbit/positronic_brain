# This file prevents importing from the legacy code repository
raise ImportError(
    "The '_historic' directory contains retired code that should not be imported directly. "
    "This code has been moved here to maintain history while we refactor the codebase. "
    "Please use the active modules in positronic_brain/ instead."
)
