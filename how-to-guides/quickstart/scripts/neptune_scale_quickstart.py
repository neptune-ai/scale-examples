import time
from random import randint, random

import numpy as np
from neptune_scale import Run

custom_id = random()  # Sets a random value for the custom run_id

run = Run(
    experiment_name="seabird-flying-skills",  # This run becomes the head of an experiment
    run_id=f"seagull-{custom_id}",  # You can customize your run_id, but if not specified, will be generated automatically
)

# Add any tags to identify your runs
run.add_tags(["Quickstart", "Long"])
run.add_tags(["Notebook"], group_tags=True)

run.log_configs(
    {
        "parameters/use_preprocessing": True,
        "parameters/learning_rate": 0.002,
        "parameters/batch_size": 64,
        "parameters/optimizer": "Adam",
        "parameters/offset": random(),
    }
)

steps = 20000  # Specify number of steps
start_time = time.time()  # Time length of execution
logging_time = 0

# Simulate the training loop
random_int = randint(0, 1000)
print(random_int)

for step in range(1, steps + 1):

    relative_progress = step / steps
    noise = np.random.uniform(-0.3, 0.3) * (1 - relative_progress)
    random_factor = np.random.uniform(0.5, 1.5)

    # Specify data to log per step as a dictionary
    metrics_to_log = {
        "train/metrics/accuracy": 1
        - 1 / np.log(relative_progress * random_int + 1.1)
        - (random() / step)
        + noise,
        "train/metrics/loss": 1 / np.log(relative_progress * random_int + 1.1)
        - (random() / step)
        + noise,
        "validation/metrics/accuracy": 1
        - 1 / np.log(relative_progress / 20 * random_int + 1.1)
        - (random() / step)
        + noise,
        "validation/metrics/loss": 1 / np.log(relative_progress / 20 * random_int + 1.1)
        - (random() / step)
        + noise,
        "test/metrics/accuracy": 1
        - 1 / np.log(relative_progress / 30 * random_int + 1.1)
        - (random() / step)
        + noise,
        "test/metrics/loss": 1 / np.log(relative_progress / 30 * random_int + 1.1)
        - (random() / step)
        + noise,
    }

    # Loop metrics to create unique values for each layer and GPU
    for i in range(1, 31):  # for each layer and each GPU
        metrics_to_log[f"debug/weights/layer_{i}"] = np.random.uniform(-0.1, 0.1)
        metrics_to_log[f"hardware/gpu_{i}"] = random()

    logging_time_start = time.time()

    # Log metrics usig the log_metrics() method
    run.log_metrics(
        data=metrics_to_log,
        step=step,
    )
    logging_time_end = time.time()
    logging_time += logging_time_end - logging_time_start

# Close run and ensure all operations are processed
run.close()

# Calculate some post run metrics for review
num_ops = steps * len(metrics_to_log)
end_time = time.time()
execution_time = end_time - start_time

print(f"Unique metrics per run: {len(metrics_to_log)}")
print(f"Number of steps per run: {steps}")
print(f"Total data points logged per run: {num_ops}")
print(
    f"Total execution time: {execution_time:.2f} seconds to process {num_ops} operations ({num_ops/execution_time:.0f} datapoints/second)."
)
print(f"Logging time {logging_time}, ({num_ops/logging_time:.0f} datapoints/second).")
