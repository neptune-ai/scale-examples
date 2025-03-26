set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running neptune_scale_quickstart.py..."
python neptune_scale_quickstart.py
