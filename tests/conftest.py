# tests/conftest.py
import sys
import os

# Ensure the root directory, the 'newsfloor' directory, and the 'newsfloor/graph' directory are in Python path.
# This makes 'config', 'node_definitions', and all newsfloor subpackages importable directly.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
newsfloor_dir = os.path.join(root_dir, "newsfloor")
graph_dir = os.path.join(newsfloor_dir, "graph")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if newsfloor_dir not in sys.path:
    sys.path.insert(0, newsfloor_dir)
if graph_dir not in sys.path:
    sys.path.insert(0, graph_dir)

