import os
from neptune_scale import Run
import torch
import glob
import re
import lm_eval

def main(run:Run):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find all run directories
    run_pattern = "./pretraining_results/*"
    run_dirs = glob.glob(run_pattern)
    
    if not run_dirs:
        print("No run directories found in ./pretraining_results/")
        return
    
    # For now, let's use the first run directory found
    # You can modify this logic if you want to process multiple runs
    run_dir = run_dirs[0]
    run_id = os.path.basename(run_dir)
    print(f"Processing run: {run_id}")
    
    # Find all checkpoint directories within this run
    checkpoint_pattern = os.path.join(run_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    # Sort checkpoints by step number
    def extract_step(checkpoint_path):
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        return int(match.group(1)) if match else 0

    checkpoint_dirs.sort(key=extract_step)

    print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate:")
    for checkpoint_dir in checkpoint_dirs:
        step = extract_step(checkpoint_dir)
        print(f"  - {checkpoint_dir} (step {step})")

    # Evaluate each checkpoint
    for checkpoint_path in checkpoint_dirs:
        global_step = extract_step(checkpoint_path)
        print(f"\nEvaluating checkpoint at step {global_step}...")
        
        try:
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={checkpoint_path},trust_remote_code=True",
                tasks=["truthfulqa_mc2", "mmlu_abstract_algebra"],
                device=device,
                num_fewshot=None, # Interate through tasks and add more shots
                limit=5,
                log_samples=True,
            )

            # Flatten the nested results and prefix with eval/benchmark
            flattened_results = {}
            for benchmark_name, benchmark_results in results["results"].items():
                for metric_name, metric_value in benchmark_results.items():
                    if metric_name != "alias":  # Skip the alias field
                        # Remove ",none" suffix from metric names
                        clean_metric_name = metric_name.replace(",none", "")
                        flattened_results[f"eval/benchmarks/{benchmark_name}/{clean_metric_name}"] = metric_value

            run.log_metrics(
                data=flattened_results,
                step=global_step
            )
            
            print(f"Successfully evaluated and logged results for step {global_step}")
            
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            continue

    print("\nEvaluation complete!")
    
if __name__ == "__main__":
    # Find the run_id from the directory structure
    run_pattern = "./pretraining_results/*"
    run_dirs = glob.glob(run_pattern)
    
    if not run_dirs:
        print("No run directories found in ./pretraining_results/")
        exit(1)
    
    # Use the first run directory found
    run_dir = run_dirs[0] # TODO: Update to account for multiple runs
    run_id = os.path.basename(run_dir)
    
    run = Run(
        experiment_name="pretraining-train",
        project="leo/pytorch-tutorial",
        run_id=run_id
    )   
    main(run)








