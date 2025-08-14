from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = load_dataset("banghua/DL-SFT-Dataset")

# Configure model and tokenizer
model_name = "./results/checkpoint-20" # "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tokenizer.pad_token = tokenizer.eos_token

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=100,
    # eval_strategy="epoch",
    # eval_steps=10,
    fp16=torch.cuda.is_available(),
    bf16=torch.cuda.is_available(),
    assistant_only_loss=False,
    completion_only_loss=False,
    # chat_template_path="HuggingFaceTB/SmolLM2-135M-Instruct"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # eval_dataset=dataset["test"],
    processing_class=tokenizer,
    # peft_config=LoRAConfig() # Add in when using LoRA
)

# Start training
trainer.train()
