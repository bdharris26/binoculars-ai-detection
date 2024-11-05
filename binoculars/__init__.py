"""
This module initializes the Binoculars package, providing the main class for AI text detection.

The `Binoculars` class is imported from the `detector` module and made available for external use.
"""

from .detector import Binoculars

__all__ = ["Binoculars"]  # Specifies the public API of the module, indicating that only the Binoculars class should be accessible when importing from this module.
