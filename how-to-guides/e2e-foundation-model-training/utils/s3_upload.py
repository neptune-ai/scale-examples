import os
import pathlib
import boto3
from transformers import TrainerCallback

__all__ = ["S3UploadLatestOnSave"]  # only export the callback

def _upload_dir_simple(local_dir: str, bucket: str, prefix: str) -> None:
    """Upload all files in local_dir to s3://bucket/prefix (flat recursive)."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            key = f"{prefix.rstrip('/')}/{rel}"
            s3.upload_file(local_path, bucket, key)

def _latest_checkpoint_dir(output_dir: str) -> str | None:
    """Return the path to the newest checkpoint-XXXX directory, or None."""
    root = pathlib.Path(output_dir)
    cands = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not cands:
        return None
    def step(p: pathlib.Path):
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1
    return str(max(cands, key=step))

class S3UploadLatestOnSave(TrainerCallback):
    """
    After each save, upload ONLY the newest 'checkpoint-XXXX' directory to:
      s3://<bucket>/<base_prefix>/checkpoint-XXXX/
    """
    def __init__(self, bucket: str, base_prefix: str, *, print_logs: bool = True):
        self.bucket = bucket
        self.base_prefix = base_prefix.rstrip("/")
        self.print_logs = print_logs
        self._uploaded_steps = set()

    @staticmethod
    def _is_main_process() -> bool:
        # Avoid duplicate uploads under DDP/Accelerate
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
        try:
            return int(rank) == 0
        except ValueError:
            return True

    def on_save(self, args, state, control, **kwargs):
        if not self._is_main_process():
            return

        ckpt = _latest_checkpoint_dir(args.output_dir)
        if not ckpt:
            return
        step_name = os.path.basename(ckpt)  # e.g. 'checkpoint-2000'
        if step_name in self._uploaded_steps:
            return  # already uploaded in this run

        dest_prefix = f"{self.base_prefix}/{step_name}"
        if self.print_logs:
            print(f"[S3] Uploading {ckpt} -> s3://{self.bucket}/{dest_prefix}/")

        _upload_dir_simple(ckpt, self.bucket, dest_prefix)
        self._uploaded_steps.add(step_name)

    def on_train_end(self, args, state, control, **kwargs):
        # Ensure the final/latest checkpoint is uploaded as well
        self.on_save(args, state, control, **kwargs)
