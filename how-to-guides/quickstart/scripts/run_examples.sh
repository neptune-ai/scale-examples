set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running neptune_scale_quickstart.py..."
curl -o sample.png https://neptune.ai/wp-content/uploads/2024/05/blog_feature_image_046799_8_3_7_3-4.jpg
python neptune_quickstart.py
