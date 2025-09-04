# Unified script for Scaling Laws and Stability Ablation Studies
# Fine-tune GPT-2 variants on WikiText-2 and analyze training behavior

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import json

import warnings
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*loss_type.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# === Experiment Configuration ===
MODEL_NAME = "gpt2"  # Choose: 'gpt2', 'gpt2-medium', etc.
MODE = "stability"      # Choose: 'scaling' or 'stability'
DEVICE = "cpu" # "auto" or "cpu"
RANDOM_WEIGHTS = False  # Set to True to initialize with random weights instead of pretrained weights

# Stability Ablation Flags (used when MODE == 'stability')
ABLATE_LAYER_NORM = True
ABLATE_DROPOUT = False
DISABLE_GRAD_CLIP = True

def main():
    # === Model and Tokenizer ===
    if RANDOM_WEIGHTS:
        # Create a model with random weights from config
        config = GPT2Config.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel(config)  # This creates a model with random weights
        model = model.to(device=DEVICE if DEVICE != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and DEVICE == "auto":
            model = model.to(torch.bfloat16)
        print(f"Initialized {MODEL_NAME} with random weights")
    else:
        # Load pre-trained model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.bfloat16)
        print(f"Loaded pre-trained {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply stability ablations
    if MODE == "stability":
        if ABLATE_LAYER_NORM:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.forward = lambda x: x
        if ABLATE_DROPOUT:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0

    # === Simple Layer Detection for smolLM ===
    def get_transformer_layers(model):
        """
        Get transformer layers for smolLM models.
        """
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = list(model.model.layers)
            print(f"Found {len(layers)} transformer layers")
            return layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Fallback for GPT-2 style models
            layers = list(model.transformer.h)
            print(f"Found {len(layers)} transformer layers (GPT-2 style)")
            return layers
        else:
            raise ValueError(f"Could not detect transformer layers for model {MODEL_NAME}")

    # === Hook-Based Metrics ===
    activations = {}
    gradient_norms = {}

    def capture_forward(layer_id):
        def hook(module, input, output):
            # Handle different output types (tensor, tuple, etc.)
            if isinstance(output, torch.Tensor):
                activations[f"metrics/train/activation/layer_{layer_id}"] = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # For models that return tuples (e.g., attention outputs)
                activations[f"metrics/train/activation/layer_{layer_id}"] = output[0]
            else:
                # Fallback: try to get the last output
                activations[f"metrics/train/activation/layer_{layer_id}"] = output
        return hook

    def capture_backward(layer_id):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                # Handle different gradient output types
                grad_tensor = grad_output[0]
                if isinstance(grad_tensor, torch.Tensor):
                    gradient_norms[f"metrics/train/grad_norms/layer_{layer_id}"] = grad_tensor.norm().item()
                else:
                    # Try to extract tensor from tuple/list
                    if isinstance(grad_tensor, (tuple, list)) and len(grad_tensor) > 0:
                        if isinstance(grad_tensor[0], torch.Tensor):
                            gradient_norms[f"metrics/train/grad_norms/layer_{layer_id}"] = grad_tensor[0].norm().item()
        return hook

    # Register hooks on detected transformer layers
    if MODE == "stability":
        transformer_layers = get_transformer_layers(model)
        for i, block in enumerate(transformer_layers):
            block.register_forward_hook(capture_forward(i))
            block.register_full_backward_hook(capture_backward(i))

    # === Dataset Prep ===
    dataset = load_dataset("roneneldan/TinyStories")
    # dataset = dataset.select(range(1000))

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    block_size = 512

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        return {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Training Arguments ===
    exp_tag = MODEL_NAME + ("_stability"+"_norm_"+str(ABLATE_LAYER_NORM)+"_dropout_"+str(ABLATE_DROPOUT)+"_grad_clip_"+str(DISABLE_GRAD_CLIP) if MODE == "stability" else "_scaling")
    training_args = TrainingArguments(
        output_dir=f"./results/{exp_tag}",
        eval_strategy="epoch",
        eval_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none",
        disable_tqdm=False,
        max_grad_norm=0.0 if (MODE == "stability" and DISABLE_GRAD_CLIP) else 1.0
    )

    # === Metrics ===
    '''def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = torch.tensor(logits[:, :-1, :])
        shift_labels = torch.tensor(labels[:, 1:])
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        perplexity = math.exp(loss.item())
        return {"perplexity": perplexity}'''

    # === Custom Callback ===
    metrics_log = {"train/gradient_norms": [], "activation_norms": []}

    class LoggingCallback(TrainerCallback):
        def __init__(self, run):
            self.run = run

        def on_log(self, args, state, control, logs=None, **kwargs):
            if MODE == "scaling":
                self.run.log_metrics(
                    data={
                        "metrics/train/loss": logs["loss"],
                        "metrics/train/grad_norm_global": logs["grad_norm"],
                        "epoch": state.epoch,
                    },
                    step=state.global_step
                )

            if MODE == "stability":
                # metrics_log["train/gradient_norms"].append(gradient_norms.copy())
                self.run.log_metrics(
                    data={
                        "metrics/train/loss": logs["loss"],
                        "metrics/train/grad_norm_global": logs["grad_norm"] if "grad_norm" in logs else 0,
                        "epoch": state.epoch,
                        **gradient_norms,
                    },
                    step=state.global_step
                )
                
                # Log activations as histograms
                if state.global_step % 5 == 0:
                    print(f"Activations: {activations}")
                    hist_dict = {}
                    for layer_id, activation in activations.items():
                        activation = activation.float()
                        flat_activation = activation.detach().cpu().flatten().numpy()
                        print(f"Flat activation: {flat_activation}, max: {flat_activation.max()}, min: {flat_activation.min()}")
                        hist, bin_edges = np.histogram(flat_activation, bins=50, range=(-3, 3))
                        hist_dict[layer_id] = Histogram(
                            bin_edges=bin_edges,
                            counts=hist,
                        )
                    print(f"Histograms: {hist_dict}")
                    self.run.log_histograms(
                        histograms=hist_dict,
                        step=state.global_step
                    )

                print(gradient_norms.copy())

        def on_evaluate(self, args, state, control, **kwargs):
            metrics_log["activation_norms"].append(activations.copy())

        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"\n[Summary] Epoch {state.epoch:.1f} complete. Checkpointing metrics...\n")

    # Neptune Logging
    from neptune_scale import Run
    from neptune_scale.types import Histogram

    run = Run(
        experiment_name=exp_tag,
        project="leo/pytorch-tutorial"
    )

    run.log_configs(
        {
            "model": MODEL_NAME,
            "mode": MODE,
            "random_weights": RANDOM_WEIGHTS,
            "ablate_layer_norm": ABLATE_LAYER_NORM,
            "ablate_dropout": ABLATE_DROPOUT,
            "disable_grad_clip": DISABLE_GRAD_CLIP,
            "sequence_length": block_size,
        }
    )

    run.add_tags(["ablations"])

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        # compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(run)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final Perplexity: {metrics['eval_perplexity']:.2f}")

    # === Save and Summarize ===
    os.makedirs(f"./results/{exp_tag}", exist_ok=True)
    with open(f"./results/{exp_tag}/metrics_log.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    # === Optional: Plot Metrics ===
    def plot_norms(norm_log, title, filename):
        steps = list(range(len(norm_log)))
        for layer_id in norm_log[0].keys():
            values = [step.get(layer_id, 0) for step in norm_log]
            plt.plot(steps, values, label=layer_id)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Norm")
        plt.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.savefig(f"./results/{exp_tag}/{filename}")
        plt.clf()

    plot_norms(metrics_log["gradient_norms"], "Gradient Norms Over Time", "gradient_norms.png")
    plot_norms(metrics_log["activation_norms"], "Activation Norms Over Time", "activation_norms.png")

    # Summary Print
    print("\n--- EXPERIMENT SUMMARY ---")
    print(f"Mode: {MODE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Final perplexity: {metrics['eval_perplexity']:.2f}")
    print(f"Logs and plots saved to ./results/{exp_tag}/")

if __name__ == "__main__":
    main()