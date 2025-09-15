import os
from pathlib import Path
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import time


def download_s3_checkpoint(s3_bucket: str, s3_path: str, local_root: str = "./checkpoints") -> str:
    """
    Download a whole checkpoint folder from S3 using a simple path string.

    Args:
        s3_bucket: S3 bucket name, e.g. "neptune-examples"
        s3_path:   Path like "transformers/run1/6000" or "transformers/run1/checkpoint-6000"
                   (function normalizes to checkpoint-<N>)
        local_root: Local root where the checkpoint dir will be placed (default: ./checkpoints)

    Returns:
        The local checkpoint directory path.

    Raises:
        FileNotFoundError if the checkpoint folder does not exist.
        ValueError for malformed inputs.
    """
    # Normalize path pieces
    s3_path = s3_path.strip().strip("/")
    parts = s3_path.split("/")
    if len(parts) < 2:
        raise ValueError("s3_path must look like 'transformers/<run_id>/<checkpoint>'")

    base_prefix = "/".join(parts[:-1])                     # e.g. transformers/run1
    ckpt_part   = parts[-1]                                # e.g. 6000 or checkpoint-6000
    if ckpt_part.startswith("checkpoint-"):
        ckpt_name = ckpt_part
    else:
        if not ckpt_part.isdigit():
            raise ValueError("checkpoint must be an integer or 'checkpoint-<N>'")
        ckpt_name = f"checkpoint-{int(ckpt_part)}"

    prefix = f"{base_prefix}/{ckpt_name}/"                 # final S3 prefix to download

    s3 = boto3.client("s3")

    # Existence check
    resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix, MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        raise FileNotFoundError(f"No objects found under s3://{s3_bucket}/{prefix}")

    # Local destination: ./outputs/<run_id>/checkpoint-XXXX/
    run_id = parts[-2]
    local_ckpt_dir = Path(local_root) / run_id / ckpt_name

    if not os.path.exists(local_ckpt_dir):
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Download all files in the checkpoint folder sequentially
        print(f"⬇️ Downloading checkpoint to: {local_ckpt_dir}")
        start_time = time.time()
        
        paginator = s3.get_paginator("list_objects_v2")
        file_count = 0
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                if not rel:   # skip folder marker keys
                    continue
                dest = local_ckpt_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(s3_bucket, key, str(dest))
                file_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"✅ Checkpoint downloaded in {elapsed_time:.2f}s ({file_count} files)")
    else:
        print(f"✅ Using cached checkpoint at {local_ckpt_dir}")

    return str(local_ckpt_dir)

def download_multiple_checkpoints_parallel(
    s3_bucket: str, 
    base_path: str, 
    checkpoints: list, 
    local_root: str = "./checkpoints", 
    max_workers: int = 5
) -> List[str]:
    """
    Download multiple checkpoint folders in parallel.
    
    Args:
        s3_bucket: S3 bucket name
        base_path: Base S3 path (e.g., "models/agile-port-20250901133956885-3rp92")
        checkpoints: List of checkpoint numbers or names
        local_root: Local root directory for downloads
        max_workers: Maximum number of parallel checkpoint downloads
        
    Returns:
        List of local checkpoint directory paths
    """
    def download_single_checkpoint(checkpoint):
        """Download a single checkpoint and return its path."""
        try:
            if isinstance(checkpoint, int):
                s3_path = f"{base_path}/checkpoint-{checkpoint}"
            else:
                s3_path = f"{base_path}/{checkpoint}"
            
            return download_s3_checkpoint(s3_bucket, s3_path, local_root)
        except Exception as e:
            print(f"❌ Failed to download checkpoint {checkpoint}: {e}")
            return None
    
    print(f"🚀 Starting parallel downloads for {len(checkpoints)} checkpoints with {max_workers} workers...")
    start_time = time.time()
    
    downloaded_paths = []
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all checkpoint download tasks
        future_to_checkpoint = {
            executor.submit(download_single_checkpoint, checkpoint): checkpoint
            for checkpoint in checkpoints
        }
        
        # Process completed downloads
        for future in as_completed(future_to_checkpoint):
            checkpoint = future_to_checkpoint[future]
            result = future.result()
            
            if result:
                downloaded_paths.append(result)
                successful_downloads += 1
                print(f"✅ Completed checkpoint {checkpoint}")
            else:
                failed_downloads += 1
                print(f"❌ Failed checkpoint {checkpoint}")
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 All downloads completed in {elapsed_time:.2f}s")
    print(f"📊 Results: {successful_downloads} successful, {failed_downloads} failed")
    
    return downloaded_paths

if __name__ == "__main__":
    # Example usage with parallel checkpoint folder downloads
    checkpoints = list(range(50, 1200, 100)) + [1171]
    
    # Download multiple checkpoint folders in parallel
    downloaded_paths = download_multiple_checkpoints_parallel(
        s3_bucket="neptune-examples",
        base_path="models/agile-port-20250901133956885-3rp92", 
        checkpoints=checkpoints,
        local_root="./pretraining_results",
        max_workers=5  # Adjust based on your bandwidth and system capabilities
    )
    
    print(f"\n📁 Downloaded {len(downloaded_paths)} checkpoint folders:")
    for path in downloaded_paths:
        print(f"  - {path}")