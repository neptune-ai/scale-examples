import os
import random
from datetime import datetime
import concurrent.futures
from typing import TypedDict, Optional
import uuid # for generating run_ids

from neptune_scale import Run
from training_simulation import *  # file that simulates training behavior without actually training a real model

API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]
PROJECT_NAME = os.environ["NEPTUNE_PROJECT"]

def get_parameters(run_index: int) -> Parameters:
    """ Get random training run parameters. """
    random.seed(run_index)

    parameters = {
        "model": {
            "batch_size": random.choice([32, 64]),
            "input_size": (1, 1024),
            "num_layers": random.choice([2, 4]),  # Reduced number of layers
            "num_heads": random.choice([4, 8]),
            "embedding_dim": random.choice([128, 256]),
            "dropout_rate": random.uniform(0.0, 0.1),
            "device": random.choice(["cpu", "cuda:0", "cuda:1"]),
            "weight_decay": random.choice([0.0, 0.01]),
            "activation_function": random.choice(["relu", "sigmoid"]),  # Added
        },
        "optimizer": {
            "lr": random.choice([1e-4, 1e-3]),
            "lr_scheduler": random.choice(["cosine", "linear"]),
            "algo": random.choice(["AdamW", "SGD"]),
        },
        "training": {
            "epochs": 1,
            "steps": 1_000,
        },
        "run_index": run_index,
    }
    return parameters


def log_nested_params(run: Run, d: dict, prefix: str = ""):
    """Logs nested config dict until it's supported natively in neptune-scale-client."""
    allowed_datatypes = [int, float, str, datetime, bool]

    flattened = {}

    def flatten(d: dict, prefix: str = ""):
        for key, value in d.items():
            new_key = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
            if isinstance(value, dict):
                flatten(d=value, prefix=new_key)
            elif type(value) in allowed_datatypes:
                flattened[new_key] = value
            else:
                flattened[new_key] = str(value)

    flatten(d)
    run.log_configs(flattened)


class RunResult(TypedDict):
    """ Information returned from each run to help in selecting forking point for next run. """
    run_id: str
    best_loss: float
    best_step: int


def execute_run(run_index, fork_run_id=None, fork_step=None) -> RunResult:
    """Executes a single training run, potentially forking from another run when optional parameters are provided."""

    experiment_name = f"Run {run_index}"  # Descriptive experiment name
    run_id = str(uuid.uuid4()) # Generate a unique run_id
    parameters = get_parameters(run_index) # Get configuration parameters for this training run

    print(f"main:INFO: {run_index=} loading")
    if fork_run_id and fork_step:
        # training from checkpoint (forking)
        run = Run(
            experiment_name=experiment_name,
            run_id=run_id,
            fork_run_id=fork_run_id,
            fork_step=fork_step,
        )

        saved_checkpoint_path = checkpoint_path(fork_run_id, fork_step)
        checkpoint = load_checkpoint(saved_checkpoint_path, parameters=parameters)

        start_step = checkpoint["step"] + 1 # start one step after the forked step
        model_state = checkpoint["model_state"]
        optimizer_state = checkpoint["optimizer_state"]

        # override checkpoint parameters with the new values where possible (no need to reinitialize weights)
        for persisted_model_key in {"num_heads", "num_layers", "input_size", "activation_function"}:
            parameters["model"][persisted_model_key] = checkpoint["parameters"]["model"][persisted_model_key]

    else:
        # training from scratch
        run = Run(
            experiment_name=experiment_name,
            run_id=run_id,
        )
        start_step = 0
        model_state = simulate_init_new_model(parameters)
        optimizer_state = simulate_init_optimizer()

    log_nested_params(run=run, d=parameters, prefix="config")

    print(f"main:INFO: {run_index=} starting")
    previous_eval_loss = 1
    best_loss = None
    best_step = None
    n_steps = parameters["training"]["steps"]
    for step_idx, step in enumerate(range(start_step, start_step + n_steps)):
        model_state, optimizer_state, loss, accuracy = simulate_training_step(model_state, optimizer_state, step, parameters)

        # log main training metrics
        run.log_metrics({
            "metrics/train/loss": loss,
            "metrics/train/accuracy": accuracy,
        }, step=step)
        
        # log debugging metrics for each layer of the model
        for layer_idx, layer_state in enumerate(model_state["layers"]):
            run.log_metrics({
                f"metrics/layer/{layer_idx:02d}/activation_mean": layer_state["activation_mean"],
                f"metrics/layer/{layer_idx:02d}/gradient_norm": layer_state["gradient_norm"]
            }, step=step)

        if step % 10 == 0:  # iterations with model evaluation
            eval_loss, eval_accuracy, eval_bleu, eval_wer = simulate_eval_step(model_state, step, parameters)

            run.log_metrics({
                "metrics/eval/loss": eval_loss,
                "metrics/eval/accuracy": eval_accuracy,
                "metrics/eval/bleu": eval_bleu,
                "metrics/eval/wer": eval_wer,
                "metrics/eval/convergence_speed": previous_eval_loss - eval_loss
            }, step=step)

            previous_eval_loss = eval_loss
            if best_loss is None or eval_loss < best_loss:
                best_loss = eval_loss
                best_step = step

            saved_checkpoint_path = save_checkpoint(run_id, step, model_state, optimizer_state, parameters)
            print(f"main:INFO: {run_index=} saved checkpoint at {saved_checkpoint_path}")
            print(f"main:INFO: {run_index=} completed {100*step_idx/n_steps:.1f}%")

    print(f"main:INFO: {run_index=} completed")
    run.close()  # flushes all reamaining buffered metrics to neptune
    return {"run_id": run_id, "best_loss": best_loss, "best_step": best_step}


def parallelize_runs_get_best_result(
        fork_num: int = 0, parent_run_id: Optional[str] = None, fork_step: Optional[int] = None
) -> RunResult:
    print(f"main:INFO: {fork_num=} runs starting")

    with concurrent.futures.ProcessPoolExecutor(max_workers=SIM_PARAMS["n_runs"]) as executor:
            futures = [
                executor.submit(
                    execute_run,
                    run_index=i + SIM_PARAMS["n_runs"] * fork_num,
                    fork_run_id=parent_run_id,
                    fork_step=fork_step,
                )
                for i in range(SIM_PARAMS["n_runs"])
            ]
            run_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            best_run = min(run_results, key=lambda rr: rr["best_loss"])
    
    print(f"main:INFO: {fork_num=} runs completed")
    return best_run


if __name__ == "__main__":
    start_timestamp = datetime.now()

    # Initial runs
    best_run = parallelize_runs_get_best_result()

    # Forking process
    for fork_num in range(1, SIM_PARAMS["n_forks"] + 1):
        best_run = parallelize_runs_get_best_result(
            fork_num, parent_run_id=best_run["run_id"], fork_step=best_run["best_step"]
        )

    end_timestamp = datetime.now()
    elapsed_time = end_timestamp - start_timestamp
    print(f"main:INFO: all forks and all runs completed in {elapsed_time=}")
