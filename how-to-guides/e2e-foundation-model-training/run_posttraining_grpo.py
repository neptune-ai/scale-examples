# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("trl-lib/tldr", split="train")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").to('cpu')
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


config = GRPOConfig(
    output_dir="./results/checkpoint-20-GRPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4, # Can set as high as 64 or 128
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=2,
    no_cuda= True,     # keeps the whole run on CPU, incl. MPS
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=config,
    train_dataset=dataset.select(range(10)),
)
trainer.train()