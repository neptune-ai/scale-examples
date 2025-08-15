# Take already existing datasets from HF
from datasets import load_dataset
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

from copy import deepcopy
from transformers import GPT2LMHeadModel, GPT2Config

from neptune_scale import Run
from utils.neptune_logger import NeptuneCallback
from utils.neptune_torchwatcher import TorchWatcher
from utils.neptune_hardware_monitoring import SystemMetricsMonitor

import warnings
warnings.filterwarnings('ignore')

def prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

def upscale_depth_prefix_suffix(
    ckpt_dir: str,
    add_k: int,
    keep_left: int = None,   # x
    keep_right: int = None,  # y
    noise_std: float = 0.01, # small noise for new blocks
) -> GPT2LMHeadModel:
    """
    Load a trained GPT-2 from `ckpt_dir`, create a deeper model with `add_k`
    blocks inserted in the middle. Copy the first `keep_left` and last `keep_right`
    blocks unchanged; initialize the `add_k` new middle blocks by copying the
    closest existing middle block (plus small noise).

    If keep_left/keep_right are None, split the old stack evenly.
    """
    base = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    cfg  = base.config
    N    = cfg.n_layer
    assert add_k > 0, "add_k must be >= 1"

    if keep_left is None or keep_right is None:
        keep_left  = N // 2
        keep_right = N - keep_left
    assert keep_left >= 0 and keep_right >= 0 and keep_left + keep_right == N, \
        "keep_left + keep_right must equal the old depth"

    new_cfg = GPT2Config(**{**cfg.to_dict(), "n_layer": N + add_k})
    big = GPT2LMHeadModel(new_cfg)

    old_blocks = list(base.transformer.h)
    new_blocks = list(big.transformer.h)

    # 1) copy prefix (first x layers)
    for i in range(keep_left):
        new_blocks[i].load_state_dict(deepcopy(old_blocks[i].state_dict()))

    # 2) insert add_k brand-new middle blocks, initialized from the "middle" donor
    donor_idx = min(keep_left, N-1) - 1 if keep_left > 0 else 0
    donor_idx = max(donor_idx, 0)
    donor = old_blocks[donor_idx]

    mid_start = keep_left
    mid_end   = keep_left + add_k
    for i in range(mid_start, mid_end):
        new_blocks[i].load_state_dict(deepcopy(donor.state_dict()))
        if noise_std and noise_std > 0:
            with torch.no_grad():
                for p in new_blocks[i].parameters():
                    p.add_(noise_std * torch.randn_like(p))

    # 3) copy suffix (last y layers) to the end of the new stack
    #    old indices [N - keep_right, ..., N-1] -> new indices [mid_end, ..., mid_end + keep_right - 1]
    for off, j in enumerate(range(N - keep_right, N)):
        new_blocks[mid_end + off].load_state_dict(deepcopy(old_blocks[j].state_dict()))

    # 4) embeddings, final LN, and tied lm_head
    big.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    big.transformer.wpe.load_state_dict(base.transformer.wpe.state_dict())
    big.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())
    big.lm_head.load_state_dict(base.lm_head.state_dict())

    return big

def calc_num_params(model):
    return sum(p.numel() for p in model.parameters())

def setup_model_and_tokenizer_and_data_collator(model_name="gpt2"):
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_name) # 12 layers, n_embd=768, 12 heads

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    return model, tokenizer, data_collator

def tokenize_function(examples, tokenizer):
    return tokenizer(
                    examples["text"], 
                    padding="max_length", # Add dynamic padding if needed
                    truncation=True,
                    max_length=128 # Truncate to 128 tokens
                )

def main(run: Run):

    model, tokenizer, data_collator = setup_model_and_tokenizer_and_data_collator(model_name="gpt2")
    '''new_model = upscale_depth_prefix_suffix("gpt2", add_k=6, keep_left=5, keep_right=7)
    print(new_model)'''
    print(calc_num_params(model))
    
    watcher = TorchWatcher(
        model,
        run,
        tensor_stats=["mean", "norm", "std", "min", "max"],  # Track mean and norm statistics
        base_namespace="model_internals",  # Default namespace for all metrics
    )

    # For testing, we'll use a small subset of the data
    # How do we split the data into train and eval?
    # Load the high-quality subset of maths related text
    data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=8)

    # Or load the larger subset
    # data = load_dataset("HuggingFaceTB/finemath", "finemath-3plus", split="train", num_proc=8)

    # print(data.features)
    raw_data = data.select(range(100))
    raw_data = raw_data.train_test_split(train_size=0.8, seed=42)
    raw_data["validation"] = raw_data.pop("test")

    tokenized_datasets = raw_data.map(lambda x: tokenize_function(x, tokenizer), 
                                batched=True,
                                remove_columns=["text", "url", "fetch_time", "content_mime_type", "warc_filename"])

    # print(tokenized_datasets["train"].features)

    training_args = TrainingArguments(
        output_dir=f"./pretraining_results/{run._run_id}",
        eval_strategy="epoch", # Evaluate at the end of each epoch
        eval_steps=1, # Evaluate every 100 steps
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        optim="adamw_torch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(), # Use mixed precision training for faster training and reduced memory usage
        gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
        logging_steps=1, # Log every 1 step
        learning_rate=2e-4, # Learning rate -> you can use a scheduler
    )

    run.log_configs(
        prefix_dict(training_args.to_dict(), "training_args"), 
        flatten=True, 
        cast_unsupported=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], # Needs to be an eval dataset
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[NeptuneCallback(run, watcher)],
    )

    trainer.train()

if __name__ == "__main__":
    
    run = Run(
        experiment_name="pretraining-gpt2",
        project="leo/pytorch-tutorial"
    )
    run.add_tags(["pretraining", "train", "e2e"])

    with SystemMetricsMonitor(run) as monitor:
        main(run)
