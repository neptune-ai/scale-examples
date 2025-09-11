from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neptune_scale import Run
from utils.neptune_logger import NeptuneCallback
from utils.neptune_torchwatcher import TorchWatcher
from utils.neptune_hardware_monitoring import SystemMetricsMonitor
from utils.s3_upload_async import S3UploadLatestOnSaveAsync

def prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

def main(run:Run):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = load_dataset("banghua/DL-SFT-Dataset")
    # dataset = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split=["math"])

    # Configure model and tokenizer
    model_name = "gpt2" #esults/checkpoint-30" #-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = tokenizer.eos_token

    # Configure trainer
    training_args = SFTConfig(
        output_dir=f"./sft_results/{run._run_id}",
        max_steps=100,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=20,
        # eval_strategy="epoch",
        # eval_steps=10,
        fp16=torch.cuda.is_available(),
        bf16=False, # Only use for training on Ampere GPUs
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

    trainer.add_callback(S3UploadLatestOnSaveAsync(
            bucket="neptune-examples", 
            base_prefix=f"models/{run._run_id}", # Save with the run id
        ))
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    run = Run(
        experiment_name="posttraining-gpt2-124M-instruct-sft",
        project="leo/pytorch-tutorial"
    )
    run.add_tags(["sft", "post-training", "e2e", "gpt2", "124M", "instruct"])
    
    with SystemMetricsMonitor(run) as monitor:
        main(run)
