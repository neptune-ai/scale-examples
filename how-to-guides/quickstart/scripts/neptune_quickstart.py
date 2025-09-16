from random import randint

import numpy as np
import requests
from neptune_scale import Run
from neptune_scale.api.run import SourceTrackingConfig
from neptune_scale.types import Histogram
from tqdm.auto import trange

NUM_STEPS = 2000  # Determines how long the training will run for
NUM_LAYERS = 10  # The theoretical number of layers to simulate


def get_gradient_norm(layer: int, step: int) -> float:
    time_decay = 1.0 / (1.0 + step / 1000)
    layer_factor = np.exp(-0.5 * ((layer - 5) ** 2) / 4)
    noise = np.random.uniform(-0.1, 0.1) * (1 - step / NUM_STEPS)

    return (0.5 + layer_factor) * time_decay + noise


def get_activation_distribution(layer: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    base_activation = np.random.normal(0, 1, 1000)
    counts, bin_edges = np.histogram(base_activation, bins=50, range=(-3, 3))

    return counts, bin_edges


def get_gpu_utilization(step: int) -> float:
    base_util = 0.85
    data_loading_drop = 0.1 if step % 100 == 0 else 0.0
    update_spike = 0.05 if step % 50 == 0 else 0.0
    noise = np.random.uniform(-0.01, 0.01)

    return (
        0 if step % (NUM_STEPS // 2) == 0 else base_util - data_loading_drop + update_spike + noise
    )


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


def download_file(url: str, filename: str) -> None:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)


def main():

    source_config = SourceTrackingConfig(
        upload_entry_point=True,
        upload_diff_head=True,
        upload_diff_upstream=True,
        upload_run_command=True,
    )

    run = Run(
        experiment_name="quickstart-experiment",
        source_tracking_config=source_config,
        project="examples/quickstart",
    )

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

    # Log a message at the beginning of the training
    run.log_string_series(
        data={
            "status": "Starting training",
        },
        step=0,
    )

    for step in range(1, NUM_STEPS):
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

        for layer in range(NUM_LAYERS):
            metrics_to_log[f"debug/gradient_norm/layer_{layer}"] = get_gradient_norm(layer, step)
            metrics_to_log[f"system/gpu_{layer}/utilization"] = get_gpu_utilization(step)

        run.log_metrics(
            data=metrics_to_log,
            step=step,
        )

        if step % 1000 == 0:
            # Log a message every 1000 steps
            run.log_string_series(
                data={
                    "status": f"Training at step {step}",
                },
                step=step,
            )

    # Log a final message
    run.log_string_series(
        data={
            "status": "Training complete!",
        },
        step=10,
    )

    download_file(
        "https://neptune.ai/wp-content/uploads/2024/05/blog_feature_image_046799_8_3_7_3-4.jpg",
        "sample.png",
    )
    download_file(
        "https://neptune.ai/wp-content/uploads/2025/05/sac-rl.mp4",
        "sac-rl.mp4",
    )
    download_file("https://neptune.ai/wp-content/uploads/2025/05/t-rex.mp3", "t-rex.mp3")

    # Upload single file to Neptune
    run.assign_files(
        {
            "files/single/image": "sample.png",
            "files/single/video": "sac-rl.mp4",
            "files/single/audio": "t-rex.mp3",
        }
    )

    # Download sample MNIST dataset
    for image_num in range(1, 10):
        try:
            response = requests.get(
                f"https://neptune.ai/wp-content/uploads/2025/05/mnist_sample_{image_num}.png"
            )
            response.raise_for_status()
            with open(f"mnist_sample_{image_num}.png", "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download mnist_sample_{image_num}.png: {e}")

    # Upload a series of files to Neptune
    for step in range(1, 10):
        run.log_files(
            files={"files/series/mnist_sample": f"mnist_sample_{step}.png"},
            step=step,
        )

    # Log custom string series
    run.log_string_series(
        data={
            "status": "Starting training",
        },
        step=0,
    )

    for step in trange(1, NUM_STEPS):

        if step % (NUM_STEPS // 2) == 0:  # Add simulated error
            run.log_string_series(
                data={
                    "status": f"Step = {step}, Loss = NaN",
                },
                step=step,
            )
        elif step % 1000 == 0:
            run.log_string_series(
                data={
                    "status": f"Step = {step}, All metrics logged",
                },
                step=step,
            )

    run.log_string_series(
        data={
            "status": "Training complete!",
        },
        step=step,
    )

    # Log series of histograms
    for step in trange(NUM_STEPS):
        hist_dict = {}  # Log every distribution at each step in a single call
        for layer in range(NUM_LAYERS):
            counts, bin_edges = get_activation_distribution(layer, step)
            activations_hist = Histogram(bin_edges=bin_edges, counts=counts)
            hist_dict[f"debug/activations/layer_{layer}"] = activations_hist

        run.log_histograms(histograms=hist_dict, step=step)

    run.close()


if __name__ == "__main__":
    main()
