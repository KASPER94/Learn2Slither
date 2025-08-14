import os
import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour les imports de tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT)) 