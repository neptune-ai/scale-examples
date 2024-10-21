set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running neptune_hpo_single_run.py..."
python neptune_hpo_single_run.py

echo "Running neptune_hpo_separate_runs.py..."
python neptune_hpo_separate_runs.py
