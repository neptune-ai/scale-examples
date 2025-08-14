# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained("./results/checkpoint-20").to('cpu')
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-20")
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(
    output_dir="./results/checkpoint-20-DPO",
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
)
trainer.train()