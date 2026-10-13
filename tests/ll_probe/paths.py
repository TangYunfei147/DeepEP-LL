"""Shared path bootstrap for directly executed probe scripts."""
import runpy
from pathlib import Path

# Do not add the source checkout to sys.path: installed-wheel tests must keep
# importing the installed DUT, including when this helper is loaded via runpy.
_paths = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/project_paths.py'))
REPO_ROOT = _paths['REPO_ROOT']
project_path = _paths['project_path']
workspace_root = _paths['workspace_root']
output_path = _paths['output_path']
results_dir = _paths['results_dir']
runtime_environment = _paths['runtime_environment']
