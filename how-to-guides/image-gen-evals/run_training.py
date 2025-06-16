import torch
import torchvision
import uuid
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from net import ClassConditionedPipeline, get_device, ClassConditionedUnet
from checkpoint_utils import (
    save_unified_checkpoint,
    create_checkpoint_path,
    load_checkpoint_by_run_and_step,
)
from neptune_scale import Run
from torchwatcher import TorchWatcher
from time import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--neptune-project", type=str, default="neptune/class-conditioned-difussion"
)
parser.add_argument("--neptune-experiment", type=str, default="debug-001")
parser.add_argument(
    "--save-checkpoints", type=bool, default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--tensor-stats",
    type=str,
    default="hist,mean,norm",
    help="Comma-separated list of tensor debug stats to track in neptune",
)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument(
    "--resume-from-run-id", type=str, default=None, help="Neptune Run ID to resume from"
)
parser.add_argument(
    "--resume-from-step", type=int, default=None, help="Global step to resume from"
)
parser.add_argument(
    "--neptune-mode",
    type=str,
    choices=["resume", "fork", "new"],
    default="new",
    help="Neptune run behavior: resume (continue existing run), fork (create forked run), new (create new run)",
)
parser.add_argument(
    "--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps"
)
parser.add_argument(
    "--ask-before-epoch",
    type=bool,
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Ask for human input before continuing to next epoch",
)



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    device = get_device()

    # Generate or use existing run ID
    if args.neptune_mode == "new" or (
        args.neptune_mode == "fork" and not args.resume_from_run_id
    ):
        run_id = str(uuid.uuid4())
        print(f"Generated new run ID: {run_id}")
    else:
        run_id = args.resume_from_run_id
        if not run_id:
            raise ValueError("run_id is required for resume/fork operations")
        print(f"Using existing run ID: {run_id}")

    # Initialize Neptune run based on mode
    if args.neptune_mode == "resume":
        print(f"Resuming Neptune run: {run_id}")
        run = Run(run_id=run_id, resume=True, project=args.neptune_project)
    elif args.neptune_mode == "fork":
        if not args.resume_from_run_id or args.resume_from_step is None:
            raise ValueError(
                "Both --resume-from-run-id and --resume-from-step are required for fork mode"
            )
        print(
            f"Forking Neptune run from {args.resume_from_run_id} at step {args.resume_from_step}"
        )
        run = Run(
            run_id=run_id,
            experiment_name=args.neptune_experiment,
            project=args.neptune_project,
            fork_run_id=args.resume_from_run_id,
            fork_step=args.resume_from_step,
        )
    else:  # new
        print(f"Creating new Neptune run: {run_id}")
        run = Run(
            run_id=run_id,
            experiment_name=args.neptune_experiment,
            project=args.neptune_project,
        )

    # Print Neptune URLs
    print(f"Neptune run URL: {run.get_run_url()}")
    if hasattr(run, "get_experiment_url"):
        print(f"Neptune experiment URL: {run.get_experiment_url()}")

    # Initialize or resume training
    if args.resume_from_run_id and args.resume_from_step is not None:
        print(
            f"Loading checkpoint from run {args.resume_from_run_id}, step {args.resume_from_step}"
        )
        try:
            pipeline, training_info = load_checkpoint_by_run_and_step(
                args.resume_from_run_id, args.resume_from_step, device=device
            )

            # Create optimizer and try to load state
            opt = torch.optim.Adam(pipeline.unet.parameters(), lr=1e-3)
            if (
                training_info
                and "optimizer_state_dict" in training_info
                and training_info["optimizer_state_dict"]
            ):
                opt.load_state_dict(training_info["optimizer_state_dict"])
                print("✓ Loaded optimizer state")
            else:
                print("! No optimizer state found, using fresh optimizer")

            net = pipeline.unet
            noise_scheduler = pipeline.scheduler
            start_epoch = training_info.get("epoch", 0)
            start_step = training_info.get("step", 0)
            start_global_step = training_info.get("global_step", args.resume_from_step)
            print(
                f"Resumed from epoch {start_epoch}, step {start_step}, global_step {start_global_step}"
            )
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch")
            net = ClassConditionedUnet().to(device)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
            )
            pipeline = ClassConditionedPipeline(unet=net, scheduler=noise_scheduler)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            start_epoch = 0
            start_step = 0
            start_global_step = 0
    else:
        print("Starting training from scratch")
        net = ClassConditionedUnet().to(device)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )
        pipeline = ClassConditionedPipeline(unet=net, scheduler=noise_scheduler)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        start_epoch = 0
        start_step = 0
        start_global_step = 0

    # Setup data
    dataset = torchvision.datasets.MNIST(
        root="mnist/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Setup logging
    watcher = TorchWatcher(
        model=net,
        run=run,
        tensor_stats=args.tensor_stats.split(","),
    )
    run.log_configs(
        {
            **{f"config/args/{k}": v for k, v in vars(args).items()},
            **{
                "config/run_id": run_id,
                "config/net/repr": str(net),
                "config/net/num_params": sum(p.numel() for p in net.parameters()),
                "config/net/num_trainable_params": sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                ),
                "config/pipeline/repr": str(pipeline),
                "config/opt/repr": str(opt),
                "config/start_epoch": start_epoch,
                "config/start_step": start_step,
                "config/start_global_step": start_global_step,
            },
        }
    )

    loss_fn = nn.MSELoss()

    # Training loop
    current_global_step = start_global_step
    for epoch in range(start_epoch, args.n_epochs):
        for step, (x, y) in enumerate(tqdm(train_dataloader)):
            # Skip steps if resuming
            if epoch == start_epoch and step < start_step:
                current_global_step += 1
                continue

            step_start = time()

            watcher.watch(
                step=current_global_step,
                track_activations=True,
                track_gradients=True,
                track_parameters=False,
            )

            # Prepare data
            x = x.to(device) * 2 - 1  # Map to (-1, 1)
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Forward pass
            pred = net(noisy_x, timesteps, y)
            loss = loss_fn(pred, noise)

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            step_end = time()

            # Log metrics
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

            # Save checkpoints
            if (
                args.save_checkpoints
                and current_global_step % args.checkpoint_interval == 0
            ):
                save_start = time()

                checkpoint_dir = create_checkpoint_path(run_id, current_global_step)

                # Save using improved unified format
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

                save_end = time()

                # Log checkpoint timing
                run.log_metrics(
                    {
                        "checkpoints/save_time_s": save_end - save_start,
                    },
                    step=current_global_step,
                )

                print(f"✓ Saved checkpoint to {checkpoint_dir}")

            current_global_step += 1

        # End of epoch
        if args.save_checkpoints:
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
        if args.ask_before_epoch:
            cont = input(f"{epoch=} completed, continue? (y/n)")
            if cont.strip().lower() == "n":
                break

    run.close()
    print("Training completed!")
