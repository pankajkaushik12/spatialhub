from .models import EfficientLoFTR
from .utils import visualize_matches

import logging

# Set up a null handler for the logger to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())
