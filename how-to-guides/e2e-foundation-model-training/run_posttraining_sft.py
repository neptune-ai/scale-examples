from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neptune_scale import Run
from utils.neptune_logger import NeptuneCallback
from utils.neptune_torchwatcher import TorchWatcher
from utils.neptune_hardware_monitoring import SystemMetricsMonitor

def prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

def main(run:Run):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = load_dataset("banghua/DL-SFT-Dataset")

    # Configure model and tokenizer
    model_name = "./results/checkpoint-30" # "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = tokenizer.eos_token

    # Configure trainer
    training_args = SFTConfig(
        output_dir=f"./post_training_sft_results/{run._run_id}",
        max_steps=30,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=10,
        # eval_strategy="epoch",
        # eval_steps=10,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        assistant_only_loss=False,
        completion_only_loss=False,
        # chat_template_path="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    '''
    run.log_configs(
        prefix_dict(training_args.to_dict(), "training_args"), 
        flatten=True, 
        cast_unsupported=True)
    '''
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        processing_class=tokenizer,
        # peft_config=LoRAConfig() # Add in when using LoRA
        callbacks=[NeptuneCallback(run)],
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    run = Run(
        experiment_name="post-training-sft",
        project="leo/pytorch-tutorial"
    )
    run.add_tags(["sft", "post-training", "e2e"])
    
    with SystemMetricsMonitor(run) as monitor:
        main(run)
