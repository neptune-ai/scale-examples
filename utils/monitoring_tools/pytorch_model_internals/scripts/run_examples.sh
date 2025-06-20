set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running torch_watcher_example.py..."
python torch_watcher_example.py
