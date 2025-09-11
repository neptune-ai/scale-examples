import os
from neptune_scale import Run
import torch
import glob
import re
import lm_eval
import time
import json
from pathlib import Path
from typing import Dict, Set, List

class CheckpointEvaluator:
    def __init__(self, results_dir: str = "./checkpoints/", check_interval: int = 60):
        """
        Initialize the checkpoint monitor.
        
        Args:
            results_dir: Directory to monitor for new runs and checkpoints
            check_interval: How often to check for new checkpoints (in seconds)
        """
        self.results_dir = results_dir
        self.check_interval = check_interval
        self.evaluated_checkpoints: Dict[str, Set[str]] = {}  # run_id -> set of evaluated checkpoint paths
        self.evaluation_log_file = "evaluation_log.json"
        self.active_runs: Dict[str, Run] = {}  # run_id -> Neptune Run object
        self.load_evaluation_log()
    
    def load_evaluation_log(self):
        """Load previously evaluated checkpoints from log file."""
        if os.path.exists(self.evaluation_log_file):
            try:
                with open(self.evaluation_log_file, 'r') as f:
                    log_data = json.load(f)
                    self.evaluated_checkpoints = {k: set(v) for k, v in log_data.items()}
                print(f"Loaded evaluation log: {len(self.evaluated_checkpoints)} runs tracked")
            except Exception as e:
                print(f"Error loading evaluation log: {e}")
    
    def save_evaluation_log(self):
        """Save evaluated checkpoints to log file."""
        try:
            log_data = {k: list(v) for k, v in self.evaluated_checkpoints.items()}
            with open(self.evaluation_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving evaluation log: {e}")
    
    def get_run_directories(self) -> List[str]:
        """Get all run directories in the results directory."""
        run_pattern = os.path.join(self.results_dir, "*")
        run_dirs = glob.glob(run_pattern)
        # Filter out files, only keep directories
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        return run_dirs
    
    def get_checkpoint_directories(self, run_dir: str) -> List[str]:
        """Get all checkpoint directories within a run directory."""
        checkpoint_pattern = os.path.join(run_dir, "checkpoint-*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        # Filter out files, only keep directories
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]
        return checkpoint_dirs
    
    def extract_step(self, checkpoint_path: str) -> int:
        """Extract step number from checkpoint path."""
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        return int(match.group(1)) if match else 0
    
    def get_or_create_run(self, run_id: str) -> Run:
        """Get existing Neptune run or create a new one for the given run_id."""
        if run_id not in self.active_runs:
            print(f"Creating new Neptune run for run_id: {run_id}")
            run = Run(
                project="leo/pytorch-tutorial",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGIyNGUwYzMtMDg2Ni00YTZlLWIyYTctZDUxN2I4ZjE5MzA1In0=",
                run_id=run_id
            )
            self.active_runs[run_id] = run
        return self.active_runs[run_id]
    
    def get_new_checkpoints(self, run_id: str, run_dir: str) -> List[str]:
        """Get checkpoints that haven't been evaluated yet."""
        checkpoint_dirs = self.get_checkpoint_directories(run_dir)
        evaluated = self.evaluated_checkpoints.get(run_id, set())
        
        new_checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir not in evaluated:
                new_checkpoints.append(checkpoint_dir)
        
        # Sort by step number
        new_checkpoints.sort(key=self.extract_step)
        return new_checkpoints
    
    def mark_checkpoint_evaluated(self, run_id: str, checkpoint_path: str):
        """Mark a checkpoint as evaluated."""
        if run_id not in self.evaluated_checkpoints:
            self.evaluated_checkpoints[run_id] = set()
        self.evaluated_checkpoints[run_id].add(checkpoint_path)
        self.save_evaluation_log()
    
    def evaluate_checkpoint(self, run: Run, checkpoint_path: str, run_id: str):
        """Evaluate a single checkpoint."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        global_step = self.extract_step(checkpoint_path)
        
        print(f"\nEvaluating checkpoint at step {global_step}...")
        print(f"Checkpoint path: {checkpoint_path}")
        
        eval_limit = 50 # Size of the evaluation set (can be updated for larger and more accurate evaluations)
        num_shots = 0 # Number of shots for the evaluation set (can be updated for larger and more accurate evaluations)
        try:
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={checkpoint_path},trust_remote_code=True",
                tasks=["truthfulqa_gen", "truthfulqa_mc1", "mmlu", "hellaswag", "arc_easy", "arithmetic", "commonsense_qa", "triviaqa", "winogrande", "gsm8k", "openbookqa"],
                device=device,
                num_fewshot=num_shots,
                limit=eval_limit,
                log_samples=True,
                batch_size=16,
            )

            # TODO: Update gsmk8 for 5 shots

            # Flatten the nested results and prefix with eval/benchmark
            flattened_results = {}
            for benchmark_name, benchmark_results in results["results"].items():
                for metric_name, metric_value in benchmark_results.items():
                    if metric_name != "alias":  # Skip the alias field
                        # Remove ",none" suffix from metric names
                        clean_metric_name = metric_name.replace(",none", "")
                        flattened_results[f"eval/benchmarks/{benchmark_name}/eval_limit_{eval_limit}/num_shots_{num_shots}/{clean_metric_name}"] = metric_value

            run.log_metrics(
                data=flattened_results,
                step=global_step
            )
            
            # Mark as evaluated
            self.mark_checkpoint_evaluated(run_id, checkpoint_path)
            print(f"Successfully evaluated and logged results for step {global_step}")
            
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            # Don't mark as evaluated if there was an error
    
    def eval_all_runs(self):
        """Monitor all run directories for new checkpoints and evaluate them."""
        run_dirs = self.get_run_directories()
        
        if not run_dirs:
            print("No run directories found.")
            return
        
        print(f"Monitoring {len(run_dirs)} run directory(ies):")
        for run_dir in run_dirs:
            run_id = os.path.basename(run_dir)
            print(f"  - {run_id}")
        
        # Process each run directory
        for run_dir in run_dirs:
            run_id = os.path.basename(run_dir)
            new_checkpoints = self.get_new_checkpoints(run_id, run_dir)
            
            if new_checkpoints:
                print(f"\nFound {len(new_checkpoints)} new checkpoints in run {run_id}:")
                for checkpoint_path in new_checkpoints:
                    step = self.extract_step(checkpoint_path)
                    print(f"  - {checkpoint_path} (step {step})")
                
                # Get or create Neptune run for this run_id
                run = self.get_or_create_run(run_id)
                
                # Evaluate each new checkpoint
                for checkpoint_path in new_checkpoints:
                    self.evaluate_checkpoint(run, checkpoint_path, run_id)
            else:
                print(f"No new checkpoints found in run {run_id}.")
    
    def run_eval_loop(self):
        """Main evaluating loop that continuously checks for new checkpoints across all runs."""
        print(f"Starting monitoring loop")
        print(f"Monitoring directory: {self.results_dir}")
        print(f"Check interval: {self.check_interval} seconds")
        print("Press Ctrl+C to stop monitoring...")
        
        try:
            while True:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new checkpoints...")
                self.eval_all_runs()
                print(f"Waiting {self.check_interval} seconds before next check...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\nEval stopped by user.")
        except Exception as e:
            print(f"Error in evaluating loop: {e}")

def main():
    """Legacy main function - now just calls the evaluator."""
    evaluator = CheckpointEvaluator()
    evaluator.run_eval_loop()

if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = CheckpointEvaluator()
    
    # Start evaluating
    evaluator.run_eval_loop()