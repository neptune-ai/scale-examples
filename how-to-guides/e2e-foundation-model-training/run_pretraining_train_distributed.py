# Take already existing datasets from HF
from datasets import load_dataset
import torch
import os

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
from utils.s3_upload_async import S3UploadLatestOnSaveAsync

import warnings
warnings.filterwarnings('ignore')

def setup_distributed():
    """Initialize distributed training for HuggingFace Trainer"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Set environment variables for HuggingFace Trainer
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        return local_rank, world_size, rank
    else:
        return 0, 1, 0

def prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

# TODO: Create a utility function to upscale the model depth
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

def setup_model_and_tokenizer_and_data_collator(model_name="gpt2", random_weights=False):
    if random_weights:
        # Create a model with random weights from config
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel(config)  # This creates a model with random weights
    else:
        # Load pre-trained model
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

def tokenize_function(examples, tokenizer, max_token_length=1024):
    return tokenizer(
                    examples["text"], 
                    padding="max_length", # Add dynamic padding if needed
                    truncation=True,
                    max_length=max_token_length # Truncate to 1024 tokens
                )

def main():
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    
    # Only create Neptune run on main process
    if local_rank == 0:
        run = Run(
            experiment_name="pretraining-gpt2-distributed",
            project="leo/pytorch-tutorial",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGIyNGUwYzMtMDg2Ni00YTZlLWIyYTctZDUxN2I4ZjE5MzA1In0=",
        )
        run.add_tags(["pretraining", "train", "e2e", "distributed"])
        run_id = run._run_id
        checkpoint_dir = f"./pretraining_results/{run_id}"
    else:
        run = None

    model, tokenizer, data_collator = setup_model_and_tokenizer_and_data_collator(model_name="gpt2", random_weights=True)
    '''new_model = upscale_depth_prefix_suffix("gpt2", add_k=6, keep_left=5, keep_right=7)
    print(new_model)'''
    if local_rank == 0:
        print(calc_num_params(model))
    
    # Setup Neptune watcher (only on main process)
    watcher = None
    if local_rank == 0 and run:
        watcher = TorchWatcher(
            model,
            run,
            tensor_stats=["mean", "norm", "std", "min", "max"],  # Track mean and norm statistics
            base_namespace="model_internals",  # Default namespace for all metrics
        )

        monitor = SystemMetricsMonitor(run)

    # For testing, we'll use a small subset of the data
    # How do we split the data into train and eval?
    # Load the high-quality subset of maths related text
    if local_rank == 0:
        print("Loading dataset...")
    data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=8)

    # Or load the larger subset
    # data = load_dataset("HuggingFaceTB/finemath", "finemath-3plus", split="train", num_proc=8)

    # print(data.features)
    raw_data = data.select(range(10000))
    raw_data = raw_data.train_test_split(train_size=0.8, seed=42)
    raw_data["validation"] = raw_data.pop("test")

    if local_rank == 0:
        print("Tokenizing dataset...")
    max_token_length = 1024
    tokenized_datasets = raw_data.map(lambda x: tokenize_function(x, tokenizer, max_token_length), 
                                batched=True,
                                remove_columns=["text", "url", "fetch_time", "content_mime_type", "warc_filename"])

    training_args = TrainingArguments(
        output_dir=checkpoint_dir if local_rank==0 else "",
        eval_strategy="steps", # Evaluate at the end of each epoch
        eval_steps=50, # Evaluate every 100 steps
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        max_steps=50,
        weight_decay=0.01,
        optim="adamw_torch",
        save_strategy="steps",  # Only save checkpoints from main process
        # save_total_limit=1,  # Keep only 1 checkpoint if saving
        fp16=torch.cuda.is_available(), # Use mixed precision training for faster training and reduced memory usage
        gradient_accumulation_steps=1, # Accumulate gradients over 2 steps
        logging_steps=1, # Log every 1 step
        learning_rate=1e-3, # Learning rate -> you can use a scheduler
        # Distributed training settings
        local_rank=local_rank,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )

    if local_rank == 0:
        run.log_configs({"config/max_token_length": max_token_length})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], # Needs to be an eval dataset
        data_collator=data_collator,
        tokenizer=tokenizer,  # Changed from processing_class to tokenizer
        callbacks=[NeptuneCallback(run, watcher)] if local_rank == 0 else [],
    )

    if local_rank == 0:
        trainer.add_callback(S3UploadLatestOnSaveAsync(
            bucket="neptune-examples", 
            base_prefix=f"models/{run_id}", # Save with the run id
        ))

    if local_rank == 0:
        monitor.start()
        
    trainer.train()

    if local_rank == 0:
        monitor.stop()

if __name__ == "__main__":

    torch.cuda.empty_cache()
    main()
