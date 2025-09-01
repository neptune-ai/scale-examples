import sys
from pathlib import Path
from typing import Literal

from neptune_scale import Run
import torch
import torchvision

from image_gen_evals.lib.net import get_device


def print_run_urls(run: Run) -> None:
    print(f"Neptune run URL: {run.get_run_url()}")
    try:
        print(f"Neptune experiment URL: {run.get_experiment_url()}")
    except ValueError:
        pass


def log_environment(run: Run, prefix: Literal["train", "eval"] = "train"):
    run.log_configs(
        {
            f"{prefix}/env/entrypoint": " ".join(sys.argv),
            f"{prefix}/env/python_version": sys.version,
            f"{prefix}/env/python_path": sys.executable,
            f"{prefix}/env/platform": sys.platform,
            f"{prefix}/env/device": get_device(),
            f"{prefix}/env/torch_version": torch.__version__,
            f"{prefix}/env/torchvision_version": torchvision.__version__,
        }
    )

    project_root = Path(__file__).parent.parent
    source_files = {
        f"{prefix}/env/source_files/{path.relative_to(project_root)}": path
        for path in project_root.rglob("*")
        if path.is_file() and not any(part.startswith(".") or part.startswith("__") for part in str(path).split("/"))
    }
    run.assign_files(source_files)
