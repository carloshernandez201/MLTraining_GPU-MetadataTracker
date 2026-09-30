import os
import sys

# Make `import profiler` work when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
