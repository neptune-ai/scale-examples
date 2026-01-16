set -e

echo "Installing requirements..."
pip install -Uq -r requirements.txt

echo "Running neptune_scale_quickstart.py..."
curl -L -o sample.png https://neptune.ai/wp-content/uploads/2024/05/blog_feature_image_046799_8_3_7_3-4.jpg
curl -L -o sac-rl.mp4 https://neptune.ai/wp-content/uploads/2025/05/sac-rl.mp4
curl -L -o t-rex.mp3  https://neptune.ai/wp-content/uploads/2025/05/t-rex.mp3

python neptune_quickstart.py
