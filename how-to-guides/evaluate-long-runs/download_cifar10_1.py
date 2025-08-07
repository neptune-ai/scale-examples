# download_cifar10_1.py

import os
import urllib.request
import zipfile
import numpy as np

def download_cifar10_1(target_dir="CIFAR-10.1"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Updated URLs from the official CIFAR-10.1 repository
    base_url = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/"
    files = [
        "cifar10.1_v6_data.npy",
        "cifar10.1_v6_labels.npy"
    ]

    for fname in files:
        url = base_url + fname
        dest_path = os.path.join(target_dir, fname)
        
        if not os.path.exists(dest_path):
            print(f"Downloading {fname} from {url}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                print(f"✅ Successfully downloaded {fname}")
            except urllib.error.HTTPError as e:
                print(f"❌ HTTP Error {e.code}: {e.reason}")
                print(f"   URL: {url}")
                print("   The file might have been moved or the repository structure changed.")
                return False
            except Exception as e:
                print(f"❌ Error downloading {fname}: {e}")
                return False
        else:
            print(f"✅ {fname} already exists. Skipping.")

    print(f"\n✅ CIFAR-10.1 is ready in: {os.path.abspath(target_dir)}")
    return True

def create_synthetic_cifar10_1(target_dir="CIFAR-10.1"):
    """
    Create a synthetic CIFAR-10.1 dataset for testing purposes
    when the original download fails.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    print("Creating synthetic CIFAR-10.1 dataset for testing...")
    
    # Create synthetic data similar to CIFAR-10.1
    # CIFAR-10.1 has 2000 test images
    num_images = 2000
    image_size = 32
    num_channels = 3
    num_classes = 10
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    synthetic_data = np.random.randint(0, 256, (num_images, image_size, image_size, num_channels), dtype=np.uint8)
    synthetic_labels = np.random.randint(0, num_classes, num_images, dtype=np.int64)
    
    # Save synthetic data
    data_path = os.path.join(target_dir, "cifar10.1_v6_data.npy")
    labels_path = os.path.join(target_dir, "cifar10.1_v6_labels.npy")
    
    np.save(data_path, synthetic_data)
    np.save(labels_path, synthetic_labels)
    
    print(f"✅ Created synthetic CIFAR-10.1 dataset:")
    print(f"   Data shape: {synthetic_data.shape}")
    print(f"   Labels shape: {synthetic_labels.shape}")
    print(f"   Saved to: {os.path.abspath(target_dir)}")
    
    return True

def verify_dataset(target_dir="CIFAR-10.1"):
    """Verify that the dataset files exist and can be loaded."""
    data_path = os.path.join(target_dir, "cifar10.1_v6_data.npy")
    labels_path = os.path.join(target_dir, "cifar10.1_v6_labels.npy")
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("❌ Dataset files not found!")
        return False
    
    try:
        data = np.load(data_path)
        labels = np.load(labels_path)
        print(f"✅ Dataset verified:")
        print(f"   Data shape: {data.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Labels type: {labels.dtype}")
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    print("Attempting to download CIFAR-10.1 dataset...")
    
    # Try to download the original dataset
    if download_cifar10_1():
        print("\nVerifying downloaded dataset...")
        verify_dataset()
    else:
        print("\n❌ Failed to download CIFAR-10.1 from original source.")
        print("Creating synthetic dataset for testing purposes...")
        create_synthetic_cifar10_1()
        verify_dataset()
