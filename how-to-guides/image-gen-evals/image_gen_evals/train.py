from typing import Annotated
import uuid
from time import perf_counter

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from neptune_scale import Run
import typer

from image_gen_evals.lib.net import ClassConditionedPipeline, get_device, ClassConditionedUnet
from image_gen_evals.lib.checkpoints import (
    save_unified_checkpoint,
    create_checkpoint_path,
    load_checkpoint_by_run_and_step,
)
from image_gen_evals.lib.torchwatcher import TorchWatcher
from image_gen_evals.lib.script_utils import print_run_urls, log_environment
from image_gen_evals.lib.neptune_hardware_monitoring import SystemMetricsMonitor



app = typer.Typer(no_args_is_help=True, add_completion=True)


def _generate_run_id() -> str:
    return str(uuid.uuid4())


def training_loop(
    *,
    run: Run,
    run_id: str,
    net: torch.nn.Module,
    pipeline: ClassConditionedPipeline,
    opt: torch.optim.Optimizer,
    start_epoch: int,
    start_step: int,
    start_global_step: int,
    save_checkpoints: bool,
    tensor_stats: list[str],
    batch_size: int,
    n_epochs: int,
    checkpoint_interval: int,
    ask_before_epoch: bool,
) -> None:
    device = get_device()
    noise_scheduler = pipeline.scheduler
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0.0, sigma=0.1)
    ])
    dataset = torchvision.datasets.MNIST(
        root=".mnist/",
        train=True,
        download=True,
        transform=transform,
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    watcher = TorchWatcher(model=net, run=run, tensor_stats=tensor_stats, base_namespace="train/debug")
    monitor = SystemMetricsMonitor(run=run, namespace="train/runtime")
    loss_fn = torch.nn.MSELoss()

    config = {
        "training_loop": {
            "save_checkpoints": save_checkpoints,
            "tensor_stats": tensor_stats,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "checkpoint_interval": checkpoint_interval,
            "ask_before_epoch": ask_before_epoch,
        },
        "net": {
            "repr": str(net),
            "num_params": sum(p.numel() for p in net.parameters()),
            "num_trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
        },
        "pipeline": {
            "repr": str(pipeline),
            "config": pipeline.config,
            "noise_scheduler": {
                "config": noise_scheduler.config,
            },
        },
        "opt": {
            "repr": str(opt),
        },
        "start_from": {
            "epoch": start_epoch,
            "step": start_step,
            "global_step": start_global_step,
        },
    }
    run.log_configs({"config": config}, flatten=True)

    current_global_step = start_global_step
    for epoch in range(start_epoch, n_epochs):
        if epoch < start_epoch:
            continue

        monitor.start()
        for step, (x, y) in enumerate(tqdm(train_dataloader)):
            if step < start_step:
                continue

            step_start = perf_counter()
            watcher.watch(
                step=current_global_step,
                track_activations=True,
                track_gradients=True,
                track_parameters=False,
            )

            x = x.to(device) * 2 - 1
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = net(noisy_x, timesteps, y)
            loss = loss_fn(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step_end = perf_counter()
            run.log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/step_time_s": step_end - step_start,
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/global_step": current_global_step,
                },
                step=current_global_step,
            )

            if save_checkpoints and current_global_step % checkpoint_interval == 0:
                save_start = perf_counter()
                checkpoint_dir = create_checkpoint_path(run_id, current_global_step)
                save_unified_checkpoint(
                    pipeline=pipeline,
                    save_directory=checkpoint_dir,
                    optimizer=opt,
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    global_step=current_global_step,
                    run_id=run_id,
                    save_hf_format=True,
                    save_pytorch_format=True,
                )
                save_end = perf_counter()

                run.log_metrics({"checkpoints/save_time_s": save_end - save_start}, step=current_global_step)
                run.log_string_series({"checkpoints/save_path": checkpoint_dir}, step=current_global_step)
                print(f"✓ Saved checkpoint to {checkpoint_dir}")

            current_global_step += 1

        monitor.stop()
        if save_checkpoints:
            final_checkpoint_dir = create_checkpoint_path(run_id, current_global_step)
            save_unified_checkpoint(
                pipeline=pipeline,
                save_directory=final_checkpoint_dir,
                optimizer=opt,
                epoch=epoch,
                step=step,
                loss=loss.item(),
                global_step=current_global_step,
                run_id=run_id,
                final_checkpoint=True,
                save_hf_format=True,
                save_pytorch_format=True,
            )
            print(f"✓ Saved final checkpoint to {final_checkpoint_dir}")

        run.wait_for_processing(timeout=60)
        if ask_before_epoch:
            cont = input(f"{epoch=} completed, continue? (y/n)")
            if cont.strip().lower() == "n":
                break

    run.close()
    print("Training completed!")


@app.command("new", help="Train a model from scratch")
def new(
    project: Annotated[str, typer.Option("--project", "-p", help="Neptune project name")],
    experiment: Annotated[str, typer.Option("--experiment", "-x", help="Experiment name for organization")],
    save_checkpoints: bool = typer.Option(True, help="Save checkpoints to a local .checkpoints directory"),
    tensor_stats: str = typer.Option(
        "hist,mean,norm",
        help="Comma-separated list of tensor debug stats to track in Neptune. Valid stats are: hist, mean, norm, min, max, var, abs_mean",
    ),
    batch_size: int = typer.Option(128, "--batch-size", "-bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "-lr", help="Learning rate"),
    n_epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    checkpoint_interval: int = typer.Option(100, "--checkpoint-interval", "-ci", help="Save checkpoint every N steps"),
    ask_before_epoch: bool = typer.Option(False, help="Ask for human input before continuing to next epoch"),
):
    run_id = _generate_run_id()
    print(f"Generated new run ID: {run_id}")
    print(f"Creating new Neptune run: {run_id}")
    run = Run(
        run_id=run_id,
        experiment_name=experiment,
        project=project,
        runtime_namespace="train/runtime",
    )
    print_run_urls(run)
    log_environment(run, prefix="train")

    device = get_device()
    net = ClassConditionedUnet().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    pipeline = ClassConditionedPipeline(unet=net, scheduler=noise_scheduler)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    tensor_stats_list = [s.strip() for s in tensor_stats.split(",") if s.strip()]
    training_loop(
        run=run,
        run_id=run_id,
        net=net,
        pipeline=pipeline,
        opt=opt,
        start_epoch=0,
        start_step=0,
        start_global_step=0,
        save_checkpoints=save_checkpoints,
        tensor_stats=tensor_stats_list,
        batch_size=batch_size,
        n_epochs=n_epochs,
        checkpoint_interval=checkpoint_interval,
        ask_before_epoch=ask_before_epoch,
    )


@app.command("resume", help="Resume training from a checkpoint, logging to the same Neptune Run")
def resume(
    project: Annotated[str, typer.Option("--project", "-p", help="Neptune project name")],
    run_id: Annotated[str, typer.Option("--run-id", "-r", help="Neptune Run ID to resume from")],
    step: Annotated[int, typer.Option("--step", "-s", help="Global step to resume from")],
    save_checkpoints: bool = typer.Option(True, help="Save checkpoints to a local .checkpoints directory"),
    tensor_stats: str = typer.Option(
        "hist,mean,norm",
        help="Comma-separated list of tensor debug stats to track in Neptune. Valid stats are: hist, mean, norm, min, max, var, abs_mean",
    ),
    batch_size: int = typer.Option(128, "--batch-size", "-bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "-lr", help="Learning rate"),
    n_epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    checkpoint_interval: int = typer.Option(100, "--checkpoint-interval", "-ci", help="Save checkpoint every N steps"),
    ask_before_epoch: bool = typer.Option(False, help="Display input prompt before each epoch"),
):
    print(f"Resuming Neptune {run_id=} at step {step=}")
    run = Run(run_id=run_id, resume=True, project=project, runtime_namespace="train/runtime")
    print_run_urls(run)
    log_environment(run)
    device = get_device()
    print(f"Loading checkpoint from run {run_id}, step {step}")
    try:
        pipeline, training_info = load_checkpoint_by_run_and_step(run_id, step, device=device)
        print("✓ Loaded checkpoint successfully")
    except Exception as e:
        raise Exception(f"Failed to load checkpoint: {e}") from e

    net = pipeline.unet
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if training_info and "optimizer_state_dict" in training_info and training_info["optimizer_state_dict"]:
        opt.load_state_dict(training_info["optimizer_state_dict"])
        print("✓ Loaded optimizer state")
    else:
        print("! No optimizer state found, using fresh optimizer")

    start_epoch = training_info.get("epoch", 0)
    start_step = training_info.get("step", 0)
    start_global_step = training_info.get("global_step", step)
    print(f"Resumed from epoch {start_epoch}, step {start_step}, global_step {start_global_step}")

    tensor_stats_list = [s.strip() for s in tensor_stats.split(",") if s.strip()]
    training_loop(
        run=run,
        run_id=run_id,
        net=net,
        pipeline=pipeline,
        opt=opt,
        start_epoch=start_epoch,
        start_step=start_step,
        start_global_step=start_global_step,
        save_checkpoints=save_checkpoints,
        tensor_stats=tensor_stats_list,
        batch_size=batch_size,
        n_epochs=n_epochs,
        checkpoint_interval=checkpoint_interval,
        ask_before_epoch=ask_before_epoch,
    )


@app.command(
    "fork",
    help="Resume training from checkpoint, potentially overriding configs, forking from the parent Neptune Run",
)
def fork(
    project: Annotated[str, typer.Option("--project", "-p", help="Neptune project name")],
    experiment: Annotated[str, typer.Option("--experiment", "-x", help="Experiment name for the new forked run")],
    parent_run_id: Annotated[str, typer.Option("--parent-run-id", "-r", help="Parent Neptune Run ID to fork from")],
    fork_step: Annotated[int, typer.Option("--fork-step", "-s", help="Global step of the parent run to fork from")],
    save_checkpoints: bool = typer.Option(True, help="Whether to save model checkpoints"),
    tensor_stats: str = typer.Option(
        "hist,mean,norm",
        help="Comma-separated list of tensor debug stats to track in Neptune. Valid stats are: hist, mean, norm, min, max, var, abs_mean",
    ),
    batch_size: int = typer.Option(128, "--batch-size", "-bs", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "-lr", help="Learning rate"),
    n_epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    checkpoint_interval: int = typer.Option(100, "--checkpoint-interval", "-ci", help="Save checkpoint every N steps"),
    ask_before_epoch: bool = typer.Option(False, help="Display input prompt before each epoch"),
):
    run_id = _generate_run_id()
    print(f"Forking {parent_run_id=} at {fork_step=} into {run_id=}")
    run = Run(
        run_id=run_id,
        experiment_name=experiment,
        project=project,
        fork_run_id=parent_run_id,
        fork_step=fork_step,
        runtime_namespace="train/runtime",
    )
    print_run_urls(run)
    log_environment(run, prefix="train")
    device = get_device()
    print(f"Loading checkpoint from run {parent_run_id}, step {fork_step}")
    try:
        pipeline, training_info = load_checkpoint_by_run_and_step(parent_run_id, fork_step, device=device)
        print("✓ Loaded checkpoint from parent run")
    except Exception as e:
        raise Exception(f"Failed to load checkpoint: {e}") from e

    net = pipeline.unet
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if training_info and "optimizer_state_dict" in training_info and training_info["optimizer_state_dict"]:
        opt.load_state_dict(training_info["optimizer_state_dict"])
        print("✓ Loaded optimizer state")
    else:
        print("! No optimizer state found, using fresh optimizer")

    start_epoch = training_info.get("epoch", 0)
    start_step = training_info.get("step", 0)
    start_global_step = training_info.get("global_step", fork_step)
    print(f"Resumed from epoch {start_epoch}, step {start_step}, global_step {start_global_step}")

    tensor_stats_list = [s.strip() for s in tensor_stats.split(",") if s.strip()]
    training_loop(
        run=run,
        run_id=run_id,
        net=net,
        pipeline=pipeline,
        opt=opt,
        start_epoch=start_epoch,
        start_step=start_step,
        start_global_step=start_global_step,
        save_checkpoints=save_checkpoints,
        tensor_stats=tensor_stats_list,
        batch_size=batch_size,
        n_epochs=n_epochs,
        checkpoint_interval=checkpoint_interval,
        ask_before_epoch=ask_before_epoch,
    )


if __name__ == "__main__":
    app()
