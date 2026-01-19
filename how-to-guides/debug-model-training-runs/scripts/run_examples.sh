set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running debug_training_runs.py..."
python debug_training_runs.py
