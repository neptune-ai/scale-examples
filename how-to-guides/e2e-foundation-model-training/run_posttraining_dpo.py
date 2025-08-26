# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from neptune_scale import Run
from utils.neptune_logger import NeptuneCallback
from utils.neptune_torchwatcher import TorchWatcher
from utils.neptune_hardware_monitoring import SystemMetricsMonitor

def prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

def main(run:Run, checkpoint_run_id:str):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = f"./post_training_sft_results/{checkpoint_run_id}/checkpoint-10"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    training_args = DPOConfig(
        output_dir=f"./post_training_dpo_results/{run._run_id}",
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=100,
        # eval_strategy="steps",
        # eval_steps=10,
        )

    trainer = DPOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset.select(range(100)),
        callbacks=[NeptuneCallback(run)],
    )
    trainer.train()

if __name__ == "__main__":
    run = Run(
        experiment_name="post-training-dpo",
        project="leo/pytorch-tutorial"
    )
    run.add_tags(["dpo", "post-training", "e2e"])
    
    # Run ID of the checkpoint we want to use for DPO (after SFT)
    checkpoint_run_id = "mathematical-fraction-20250814164524341-bjqix" 

    with SystemMetricsMonitor(run) as monitor:
        main(run, checkpoint_run_id)