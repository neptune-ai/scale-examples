from random import randint
from typing import Literal

import numpy as np
from neptune_scale import Run

NUM_STEPS = 2000  # Determines how long the training will run for


def get_gradient_norm(layer: int, step: int) -> float:
    time_decay = 1.0 / (1.0 + step / 1000)
    layer_factor = np.exp(-0.5 * ((layer - 5) ** 2) / 4)
    noise = np.random.uniform(-0.1, 0.1) * (1 - step / NUM_STEPS)

    return (0.5 + layer_factor) * time_decay + noise


def get_gpu_utilization(step: int) -> float:
    base_util = 0.85
    data_loading_drop = 0.2 if step % 10 == 0 else 0.0
    update_spike = 0.1 if step % 5 == 0 else 0.0
    noise = np.random.uniform(-0.05, 0.05)

    return base_util - data_loading_drop + update_spike + noise


def _generate_metric(
    step: int,
    factor: float = 1.0,
) -> float:
    relative_progress = step / NUM_STEPS
    noise = np.random.uniform(-0.3, 0.3) * (1 - relative_progress)
    random_int = randint(0, 1000)

    return 1 / np.log(relative_progress / factor * random_int + 1.1) + noise


def training_step(step: int) -> tuple[float, float]:
    accuracy = 0.45 + 1 / (1 + np.exp(_generate_metric(step)))
    loss = _generate_metric(step)
    return accuracy, loss


def validation_step(step: int) -> tuple[float, float]:
    accuracy = 0.45 + 1 / (1 + np.exp(_generate_metric(step, 20)))
    loss = _generate_metric(step, 20)
    return accuracy, loss


def test_step(step: int) -> tuple[float, float]:
    accuracy = 0.45 + 1 / (1 + np.exp(_generate_metric(step, 30)))
    loss = _generate_metric(step, 30)
    return accuracy, loss


def main():
    run = Run(experiment_name="quickstart-experiment")

    print(f"Neptune run created 🎉\nAccess at {run.get_run_url()}")

    run.add_tags(["quickstart", "script"])
    run.add_tags(["short"], group_tags=True)

    run.log_configs(
        {
            "parameters/data/use_preprocessing": True,
            "parameters/data/batch_size": 64,
            "parameters/model/activation": "relu",
            "parameters/model/dropout": 0.2,
            "parameters/optimizer/type": "Adam",
            "parameters/optimizer/learning_rate": 0.002,
        }
    )

    for step in range(NUM_STEPS):
        train_accuracy, train_loss = training_step(step)
        valid_accuracy, valid_loss = validation_step(step)
        test_accuracy, test_loss = test_step(step)

        metrics_to_log = {
            "metrics/train/accuracy": train_accuracy,
            "metrics/train/loss": train_loss,
            "metrics/valid/accuracy": valid_accuracy,
            "metrics/valid/loss": valid_loss,
            "metrics/test/accuracy": test_accuracy,
            "metrics/test/loss": test_loss,
        }

        for layer in range(10):
            metrics_to_log[f"debug/gradient_norm/layer_{layer}"] = get_gradient_norm(layer, step)
            metrics_to_log[f"system/gpu_{layer}/utilization"] = get_gpu_utilization(step)

        run.log_metrics(
            data=metrics_to_log,
            step=step,
        )

    # Upload single file to Neptune
    run.assign_files({"files/single_image": "sample.png"})

    # Log custom string series
    for step in range(1, 10):

        run.log_string_series(
            data={
                "custom_messages/errors": f"Job failed - step {step}",
                "custom_messages/info": f"Training completed - step {step}",
            },
            step=step,
        )

    run.close()


if __name__ == "__main__":
    main()
