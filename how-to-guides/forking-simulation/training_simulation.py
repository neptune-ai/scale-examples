import os
import random
import math
from datetime import datetime
import tempfile
from time import sleep
import json  # For saving and loading checkpoints
from typing import Literal, Tuple, TypedDict

# --- Simulation Parameters ---
SIM_PARAMS = {
    "n_runs": 5,
    "checkpoint_interval": 100,
    "n_forks": 10,
    "eval_sleep": 0,
    "flatline_probability": 0.35,
    "fast_convergence_probability": 0.1,
    "sudden_divergence_probability": 0.35,
    # Reduced spike probability, especially for training
    "spike_probability": 0.15,
}

# --- Realism Parameters ---
# Noise levels (Eval > Train) - Increased training noise slightly again
TRAIN_LOSS_NOISE = 0.05
TRAIN_ACC_NOISE = 0.025
EVAL_LOSS_NOISE = 0.10 # Increased eval noise
EVAL_ACC_NOISE = 0.04
EVAL_BLEU_NOISE = 0.04
EVAL_WER_NOISE = 0.03

# Oscillation Parameters (Amplitude & Frequency)
# Drastically reduced for training, kept for eval
TRAIN_LOSS_OSC_AMP = 0.01 # Minimal oscillation for train loss
TRAIN_ACC_OSC_AMP = 0.005 # Minimal oscillation for train accuracy
EVAL_LOSS_OSC_AMP = 0.15 # Keep eval oscillations
EVAL_ACC_OSC_AMP = 0.05
EVAL_BLEU_OSC_AMP = 0.05
EVAL_WER_OSC_AMP = 0.04
OSC_FREQ_FACTOR = 5.0 # Slightly lower frequency

# Convergence Speed & Plateau Parameters
BASE_DECAY_FACTOR_NORMAL = 1.0
BASE_DECAY_FACTOR_FAST = 0.6
CONVERGENCE_POWER = 0.7

# Transition speed factor for divergence/flatline
TRANSITION_K = 5.0

# Spike magnitude parameters (smaller for training)
TRAIN_SPIKE_LOSS_DELTA_RANGE = (0.1, 0.4)
TRAIN_SPIKE_ACC_DELTA_RANGE = (-0.04, -0.01)
EVAL_SPIKE_LOSS_DELTA_RANGE = (0.4, 1.0)
EVAL_SPIKE_ACC_DELTA_RANGE = (-0.1, -0.04)


# --- Type Definitions ---
class ModelConfig(TypedDict):
    batch_size: int
    input_size: Tuple[int, int]
    num_layers: int
    num_heads: int
    embedding_dim: int
    dropout_rate: float
    device: Literal["cpu"]
    weight_decay: float
    activation_function: Literal["relu", "sigmoid"]

class OptimizerConfig(TypedDict):
    lr: float
    lr_scheduler: Literal["cosine", "linear"]
    algo: Literal["AdamW", "SGD"]

class TrainingConfig(TypedDict):
    epochs: int
    steps: int

class Parameters(TypedDict):
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    run_index: int


def get_parameters(run_index: int, n_steps: int = 1_000) -> Parameters:
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
            "steps": n_steps,
        },
        "run_index": run_index,
    }
    return parameters


class LayerState(TypedDict):
    activation_mean: float
    gradient_norm: float

class ModelState(TypedDict):
    simulation_behavior: Literal["sudden_divergence", "flatline", "normal", "fast_convergence"]
    simulation_behavior_start_step: int
    layers: list[LayerState]
    activation_function: Literal["relu", "sigmoid"]
    has_spike: bool
    spike_step: int
    spike_loss_delta: float # Magnitude for the spike
    spike_acc_delta: float # Magnitude for the spike

class OptimizerState(TypedDict):
    prev_train_loss: float
    prev_train_acc: float
    best_train_loss: float
    best_train_acc: float

# --- Checkpointing ---
def checkpoint_path(run_id: str, step: int) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    return f"checkpoints/checkpoint_{run_id}_step_{step}.json"

def save_checkpoint(run_id: str, step: int, model_state: ModelState, optimizer_state: OptimizerState, parameters: Parameters) -> str:
    # Prune large unserializable objects if necessary before saving
    # For this simulation, it's likely fine, but good practice for real models
    checkpoint_data = {
        "step": step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "parameters": parameters
    }
    saved_checkpoint_path = checkpoint_path(run_id, step)
    temp_path = saved_checkpoint_path + ".tmp"
    try:
        with open(temp_path, 'w') as tmp_file:
            json.dump(checkpoint_data, tmp_file, indent=2) # Add indent for readability
        os.replace(temp_path, saved_checkpoint_path)
    except TypeError as e:
        print(f"Serialization error saving checkpoint: {e}")
        # Handle non-serializable data if it occurs
        # For example, convert numpy arrays or other objects to lists/basic types
        raise
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return saved_checkpoint_path


def load_checkpoint(checkpoint_path: str, parameters: Parameters):
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    # Reset behavior and spike parameters for the forked run
    checkpoint["model_state"]["simulation_behavior"] = _simulate_run_behavior()
    checkpoint["model_state"]["simulation_behavior_start_step"] = _simulate_run_behavior_start_step(parameters=parameters)
    checkpoint["model_state"] = _initialize_spike_params(checkpoint["model_state"], parameters)
    return checkpoint

# --- Simulation Behavior Logic ---
def _simulate_run_behavior():
    # Ensure probabilities sum <= 1
    flat_prob = SIM_PARAMS["flatline_probability"]
    fast_prob = SIM_PARAMS["fast_convergence_probability"]
    div_prob = SIM_PARAMS["sudden_divergence_probability"]
    norm_prob = max(0, 1.0 - flat_prob - fast_prob - div_prob)

    behavior_probabilities = {
        "normal": norm_prob,
        "flatline": flat_prob,
        "fast_convergence": fast_prob,
        "sudden_divergence": div_prob
    }
    total_probability = sum(behavior_probabilities.values())
    if total_probability <= 0: return "normal" # Fallback
    # Normalize if needed (due to max(0, ...))
    behavior_probabilities = {k: v / total_probability for k, v in behavior_probabilities.items()}

    return random.choices(list(behavior_probabilities.keys()), weights=list(behavior_probabilities.values()), k=1)[0]

def _simulate_run_behavior_start_step(parameters: dict) -> int:
    total_steps = parameters["training"]["steps"]
    min_step = total_steps // 2
    max_step = max(min_step + 1, total_steps - 100)
    beta_value = random.betavariate(2, 1) # Bias towards later steps
    return int(min_step + (max_step - min_step) * beta_value)

def _initialize_spike_params(model_state: ModelState, parameters: Parameters) -> ModelState:
    model_state["has_spike"] = False
    model_state["spike_step"] = -1
    model_state["spike_loss_delta"] = 0.0
    model_state["spike_acc_delta"] = 0.0
    # Spikes are less likely now overall
    if model_state["simulation_behavior"] in ["sudden_divergence", "flatline"]:
        if random.random() < SIM_PARAMS["spike_probability"]:
            total_steps = parameters["training"]["steps"]
            behavior_start = model_state["simulation_behavior_start_step"]
            min_spike_step = behavior_start + 50
            max_spike_step = total_steps - 50
            if min_spike_step < max_spike_step:
                 model_state["has_spike"] = True
                 model_state["spike_step"] = random.randint(min_spike_step, max_spike_step)
                 # Use the specific ranges for spike magnitude
                 model_state["spike_loss_delta"] = random.uniform(*EVAL_SPIKE_LOSS_DELTA_RANGE) # Eval uses larger range by default
                 model_state["spike_acc_delta"] = random.uniform(*EVAL_SPIKE_ACC_DELTA_RANGE)
    return model_state

# --- Initialization ---
def simulate_init_new_model(parameters: dict) -> ModelState:
    num_layers = parameters["model"]["num_layers"]
    activation_function = parameters["model"]["activation_function"]
    model_state: ModelState = {
        "layers": [{"activation_mean": 0.0, "gradient_norm": 0.0} for _ in range(num_layers)],
        "simulation_behavior": _simulate_run_behavior(),
        "simulation_behavior_start_step": _simulate_run_behavior_start_step(parameters=parameters),
        "activation_function": activation_function,
        "has_spike": False, "spike_step": -1, "spike_loss_delta": 0.0, "spike_acc_delta": 0.0,
    }
    model_state = _initialize_spike_params(model_state, parameters)
    return model_state

def simulate_init_optimizer() -> OptimizerState:
    return {
        "prev_train_loss": random.uniform(2.8, 3.2), # Start higher
        "prev_train_acc": random.uniform(0.04, 0.08), # Start lower
        "best_train_loss": float('inf'),
        "best_train_acc": 0.0,
    }

# --- Layer Metric Simulation ---
def _simulate_layer_metrics(model_state: ModelState, step: int, parameters: Parameters) -> ModelState:
    # Keep layer metrics relatively smooth, slow decay
    num_layers = len(model_state["layers"])
    activation_function = model_state["activation_function"]
    total_steps = parameters["training"]["steps"]
    progress = step / total_steps

    for layer_idx in range(num_layers):
        decay_factor = max(0.2, math.exp(-(progress / BASE_DECAY_FACTOR_NORMAL)**CONVERGENCE_POWER))
        oscillation = 0.005 * math.sin(OSC_FREQ_FACTOR * math.pi * progress + layer_idx) # Very small oscillation

        if activation_function == "relu":
            activation_mean = 0.15 * decay_factor + 0.05 + oscillation + random.gauss(0, 0.01)
            gradient_norm = 0.025 * decay_factor + 0.005 + abs(oscillation*0.5) + random.gauss(0, 0.003)
        else: # sigmoid
            activation_mean = 0.5 + 0.05 * decay_factor + oscillation + random.gauss(0, 0.01)
            gradient_norm = 0.012 * decay_factor + 0.003 + abs(oscillation*0.5) + random.gauss(0, 0.002)

        model_state["layers"][layer_idx] = {
            "activation_mean": max(0, activation_mean),
            "gradient_norm": max(0, gradient_norm)
        }
    return model_state

# --- Base Curve Definitions (Smoother Training, Noisy Eval) ---
def _get_progress(step, total_steps):
    # Clamp progress to avoid issues near step 0 or total_steps
    return max(0.0, min(1.0, step / total_steps))

def _get_decay(progress, decay_factor):
    # Ensure progress is slightly > 0 for the power calculation if needed
    safe_progress = max(1e-6, progress)
    return math.exp(-(safe_progress / decay_factor)**CONVERGENCE_POWER)

def _get_oscillation(progress, amplitude, freq_factor):
    return amplitude * math.sin(freq_factor * math.pi * progress)

# Normal Convergence (Smoother Training)
def _base_loss_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, TRAIN_LOSS_OSC_AMP, OSC_FREQ_FACTOR) # Minimal osc
    # Initial ~3.0, Final ~1.8. Clearer downward trend.
    return 1.8 + 1.2 * decay + osc

def _base_acc_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, TRAIN_ACC_OSC_AMP, OSC_FREQ_FACTOR) # Minimal osc
    # Initial ~0.06, Final ~0.31. Clearer upward trend.
    return 0.06 + 0.25 * (1 - decay) + osc

# Fast Convergence (Smoother Training)
def _base_loss_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST) # Faster decay
    osc = _get_oscillation(progress, TRAIN_LOSS_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~3.0, Final ~1.6
    return 1.6 + 1.4 * decay + osc

def _base_acc_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST) # Faster decay
    osc = _get_oscillation(progress, TRAIN_ACC_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.06, Final ~0.36
    return 0.06 + 0.30 * (1 - decay) + osc

def _smooth_transition_factor(step, start_step, total_steps, k=TRANSITION_K):
    if step <= start_step: return 0.0
    denominator = total_steps - start_step
    if denominator <= 0: return 1.0
    progress = max(0.0, (step - start_step) / denominator)
    # Use a sigmoid-like function for smoother start/end of transition
    return 1 / (1 + math.exp(-k * (progress - 0.5))) # Centered around 0.5 progress


# --- Training Step Simulation ---
def simulate_training_step(
        model_state: ModelState, optimizer_state: OptimizerState, step: int, parameters: Parameters
) -> Tuple[ModelState, OptimizerState, float, float]:
    """Simulates training loss/accuracy: smoother, converges somewhat, small spikes."""
    behavior = model_state["simulation_behavior"]
    behavior_start = model_state["simulation_behavior_start_step"]
    total_steps = parameters["training"]["steps"]

    # Determine base curve (now much smoother)
    if behavior == "fast_convergence":
        base_loss = _base_loss_fast(step, total_steps)
        base_accuracy = _base_acc_fast(step, total_steps)
    else: # Normal or base for divergence/flatline
        base_loss = _base_loss_normal(step, total_steps)
        base_accuracy = _base_acc_normal(step, total_steps)

    # Apply smooth transition for divergence or flatline (less dramatic targets for training)
    if step >= behavior_start:
        transition = _smooth_transition_factor(step, behavior_start, total_steps)
        loss_at_start = _base_loss_normal(behavior_start, total_steps) # Base value before transition
        acc_at_start = _base_acc_normal(behavior_start, total_steps) # Base value before transition

        if behavior == "sudden_divergence":
            # Training divergence: Loss increases moderately, accuracy drops moderately
            target_loss = loss_at_start + 0.8
            target_acc = max(0.02, acc_at_start - 0.15) # Don't drop below minimum
            base_loss = loss_at_start + (target_loss - loss_at_start) * transition
            base_accuracy = acc_at_start + (target_acc - acc_at_start) * transition
        elif behavior == "flatline":
            # Training flatline: Stagnates very close to where it was
            target_loss = loss_at_start + 0.05
            target_acc = max(0.02, acc_at_start - 0.02)
            base_loss = loss_at_start + (target_loss - loss_at_start) * transition
            base_accuracy = acc_at_start + (target_acc - acc_at_start) * transition

    # Add noise (moderate level for training)
    loss = base_loss + random.gauss(0, TRAIN_LOSS_NOISE)
    accuracy = base_accuracy + random.gauss(0, TRAIN_ACC_NOISE)

    # Apply SMALL spike if applicable (only for training step)
    if model_state["has_spike"] and step == model_state["spike_step"]:
        # Use smaller training spike ranges
        loss += random.uniform(*TRAIN_SPIKE_LOSS_DELTA_RANGE)
        accuracy += random.uniform(*TRAIN_SPIKE_ACC_DELTA_RANGE)
        # No need to modify model_state spike values here, they are for eval by default

    # Ensure metrics stay in reasonable ranges
    loss = max(0.1, loss)
    accuracy = max(0.0, min(0.95, accuracy))

    # Update layer metrics (kept smooth)
    model_state = _simulate_layer_metrics(model_state=model_state, step=step, parameters=parameters)

    # Update optimizer state
    best_acc = max(optimizer_state["best_train_acc"], accuracy)
    best_loss = min(optimizer_state["best_train_loss"], loss)
    optimizer_state = {
        "best_train_acc": best_acc, "best_train_loss": best_loss,
        "prev_train_acc": accuracy, "prev_train_loss": loss,
    }
    return model_state, optimizer_state, loss, accuracy

# --- Base Eval Curve Definitions (Noisy, Oscillating) ---
# Keep these similar to the previous version to maintain noisy eval look
def _base_eval_loss_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, EVAL_LOSS_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~3.2, Final ~2.2
    return 2.2 + 1.0 * decay + osc

def _base_eval_acc_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, EVAL_ACC_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.08, Final ~0.28
    return 0.08 + 0.20 * (1 - decay) + osc

def _base_eval_bleu_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, EVAL_BLEU_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.04, Final ~0.24
    return 0.04 + 0.20 * (1 - decay) + osc

def _base_eval_wer_normal(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_NORMAL)
    osc = _get_oscillation(progress, EVAL_WER_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.95, Final ~0.7
    return 0.7 + 0.25 * decay - osc # Higher WER is worse

def _base_eval_loss_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST)
    osc = _get_oscillation(progress, EVAL_LOSS_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~3.2, Final ~1.9
    return 1.9 + 1.3 * decay + osc

def _base_eval_acc_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST)
    osc = _get_oscillation(progress, EVAL_ACC_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.08, Final ~0.38
    return 0.08 + 0.30 * (1 - decay) + osc

def _base_eval_bleu_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST)
    osc = _get_oscillation(progress, EVAL_BLEU_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.04, Final ~0.34
    return 0.04 + 0.30 * (1 - decay) + osc

def _base_eval_wer_fast(step, total_steps):
    progress = _get_progress(step, total_steps)
    decay = _get_decay(progress, BASE_DECAY_FACTOR_FAST)
    osc = _get_oscillation(progress, EVAL_WER_OSC_AMP, OSC_FREQ_FACTOR)
    # Initial ~0.95, Final ~0.6
    return 0.6 + 0.35 * decay - osc

# --- Evaluation Step Simulation ---
def simulate_eval_step(model_state: ModelState, step: int, parameters: Parameters) -> Tuple[float, float, float, float]:
    """ Simulates evaluation: noisy, oscillating, potentially divergent/flatlining."""
    sleep(SIM_PARAMS["eval_sleep"])

    behavior = model_state["simulation_behavior"]
    behavior_start = model_state["simulation_behavior_start_step"]
    total_steps = parameters["training"]["steps"]

    # Calculate base values (noisy, oscillating)
    if behavior == "fast_convergence":
        base_eval_loss = _base_eval_loss_fast(step, total_steps)
        base_eval_accuracy = _base_eval_acc_fast(step, total_steps)
        base_eval_bleu = _base_eval_bleu_fast(step, total_steps)
        base_eval_wer = _base_eval_wer_fast(step, total_steps)
    else: # Normal or base for divergence/flatline
        base_eval_loss = _base_eval_loss_normal(step, total_steps)
        base_eval_accuracy = _base_eval_acc_normal(step, total_steps)
        base_eval_bleu = _base_eval_bleu_normal(step, total_steps)
        base_eval_wer = _base_eval_wer_normal(step, total_steps)

    # Apply transitions for divergence/flatline (more dramatic targets for eval)
    if step >= behavior_start:
        transition = _smooth_transition_factor(step, behavior_start, total_steps)
        loss_at_start = _base_eval_loss_normal(behavior_start, total_steps)
        acc_at_start = _base_eval_acc_normal(behavior_start, total_steps)
        bleu_at_start = _base_eval_bleu_normal(behavior_start, total_steps)
        wer_at_start = _base_eval_wer_normal(behavior_start, total_steps)

        if behavior == "sudden_divergence":
            # Eval divergence: Loss explodes, others plummet
            target_loss = loss_at_start + 2.5
            target_acc = 0.01
            target_bleu = 0.01
            target_wer = 0.98
            base_eval_loss = loss_at_start + (target_loss - loss_at_start) * transition
            base_eval_accuracy = acc_at_start + (target_acc - acc_at_start) * transition
            base_eval_bleu = bleu_at_start + (target_bleu - bleu_at_start) * transition
            base_eval_wer = wer_at_start + (target_wer - wer_at_start) * transition
        elif behavior == "flatline":
            # Eval flatline: Stagnates at poor values
            target_loss = loss_at_start + 0.3
            target_acc = max(0.01, acc_at_start - 0.05)
            target_bleu = max(0.01, bleu_at_start - 0.05)
            target_wer = min(0.99, wer_at_start + 0.1)
            base_eval_loss = loss_at_start + (target_loss - loss_at_start) * transition
            base_eval_accuracy = acc_at_start + (target_acc - acc_at_start) * transition
            base_eval_bleu = bleu_at_start + (target_bleu - bleu_at_start) * transition
            base_eval_wer = wer_at_start + (target_wer - wer_at_start) * transition

    # Add noise (higher level for eval)
    eval_loss = base_eval_loss + random.gauss(0, EVAL_LOSS_NOISE)
    eval_accuracy = base_eval_accuracy + random.gauss(0, EVAL_ACC_NOISE)
    eval_bleu = base_eval_bleu + random.gauss(0, EVAL_BLEU_NOISE)
    eval_wer = base_eval_wer + random.gauss(0, EVAL_WER_NOISE)

    # Apply LARGER spike if applicable (using model_state values directly)
    if model_state["has_spike"] and step == model_state["spike_step"]:
        eval_loss += model_state["spike_loss_delta"] # Use the larger eval spike delta
        eval_accuracy += model_state["spike_acc_delta"]
        # Apply proportional spikes to BLEU/WER
        eval_bleu += model_state["spike_acc_delta"] # Lower BLEU is bad
        eval_wer -= model_state["spike_acc_delta"] * 0.8 # Lower WER is good, maybe slightly less impact

    # Ensure metrics stay in reasonable ranges
    eval_loss = max(0.1, eval_loss)
    eval_accuracy = max(0.0, min(0.95, eval_accuracy))
    eval_bleu = max(0.0, min(0.95, eval_bleu))
    eval_wer = max(0.01, min(1.0, eval_wer))

    return eval_loss, eval_accuracy, eval_bleu, eval_wer
