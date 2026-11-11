"""
Pytest Configuration & Fixture Initializer.
Adds project root directory to sys.path so modules like `utils` and `agents` import seamlessly.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
