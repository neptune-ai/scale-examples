import os, pathlib, mimetypes
from queue import Queue
from threading import Thread
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from transformers import TrainerCallback

def _content_type(p: str) -> str:
    import mimetypes
    c, _ = mimetypes.guess_type(p); return c or "application/octet-stream"

def _upload_dir(local_dir: str, bucket: str, prefix: str, *, sse=None):
    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    tcfg = TransferConfig(use_threads=True, max_concurrency=16, multipart_threshold=8*1024**2, multipart_chunksize=8*1024**2)
    for root, _, files in os.walk(local_dir):
        for name in files:
            lp = os.path.join(root, name)
            rel = os.path.relpath(lp, local_dir).replace(os.sep, "/")
            key = f"{prefix.rstrip('/')}/{rel}"
            extra = {"ContentType": _content_type(lp)}
            if sse: extra["ServerSideEncryption"] = sse
            s3.upload_file(lp, bucket, key, ExtraArgs=extra, Config=tcfg)

def _latest_ckpt_dir(output_dir: str) -> str | None:
    root = pathlib.Path(output_dir)
    cands = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not cands: return None
    def step(p): 
        try: return int(p.name.split("-")[-1])
        except: return -1
    return str(max(cands, key=step))

class _AsyncUploader:
    def __init__(self, bucket: str, base_prefix: str, queue_size: int = 4, sse: str | None = None):
        self.bucket = bucket
        self.base_prefix = base_prefix.rstrip("/")
        self.sse = sse
        self.q = Queue(maxsize=queue_size)
        self.t = Thread(target=self._run, daemon=True)
        self.t.start()
    def _run(self):
        while True:
            item = self.q.get()
            if item is None: break
            local_dir = item
            name = os.path.basename(local_dir.rstrip("/"))
            prefix = f"{self.base_prefix}/{name}" if self.base_prefix else name
            print(f"[S3] (async) Uploading {local_dir} -> s3://{self.bucket}/{prefix}/")
            _upload_dir(local_dir, self.bucket, prefix, sse=self.sse)
            self.q.task_done()
    def enqueue(self, local_dir: str):
        self.q.put(local_dir)  # blocks briefly if queue is full (backpressure)
    def close(self):
        self.q.put(None); self.t.join()

class S3UploadLatestOnSaveAsync(TrainerCallback):
    """Enqueue the newest checkpoint dir for background upload (threaded)."""
    def __init__(self, bucket: str, base_prefix: str, *, sse: str | None = None):
        self.bucket = bucket
        self.base_prefix = base_prefix.rstrip("/")
        self.sse = sse
        self.uploader = _AsyncUploader(bucket, self.base_prefix, sse=sse)
        self._uploaded = set()
    @staticmethod
    def _is_main_process() -> bool:
        r = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
        try: return int(r) == 0
        except ValueError: return True
    def on_save(self, args, state, control, **kwargs):
        if not self._is_main_process(): return
        ckpt = _latest_ckpt_dir(args.output_dir)
        if not ckpt: return
        name = os.path.basename(ckpt)
        if name in self._uploaded: return
        self.uploader.enqueue(ckpt)
        self._uploaded.add(name)
    def on_train_end(self, args, state, control, **kwargs):
        # Finish any pending uploads before exiting
        self.uploader.close()
