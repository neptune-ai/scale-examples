import os
import time
import tempfile
import torch
import statistics
from typing import Annotated
from image_gen_evals.lib.checkpoints import load_checkpoint_by_run_and_step
from image_gen_evals.lib.net import get_device
from google import genai
from google.genai.types import File
from neptune_scale import Run
import typer
from image_gen_evals.lib.script_utils import print_run_urls, log_environment


app = typer.Typer(no_args_is_help=True, add_completion=True)


class GeminiEvaluator:
    def __init__(self, model_name: str, requests_per_minute: int) -> "GeminiEvaluator":
        """Initialize Gemini evaluator with rate limiting."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        assert self.api_key, "GEMINI_API_KEY env variable must be set"

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.sleep_time = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = time.time()
        print(f"Initialized {self}")

    def __repr__(self):
        return f"GeminiEvaluator(model_name={self.model_name}, sleep_time={self.sleep_time:.2f}s)"

    def upload_image_once(self, image_path: str) -> File:
        """Upload image and return file object."""
        self.wait_for_rate_limit()
        result = self.client.files.upload(file=image_path)
        self.last_request_time = time.time()
        return result

    def evaluate(self, file: File, prompt: str, system_prompt: str) -> str:
        """Evaluate image with Gemini model."""
        self.wait_for_rate_limit()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                file,
                f"{system_prompt}\n{prompt}",
            ],  # not all models support configured system prompt
        )
        result = response.text.strip()
        self.last_request_time = time.time()
        return result

    def wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        now = time.time()
        if now - self.last_request_time < self.sleep_time:
            time.sleep(self.sleep_time - (now - self.last_request_time))


@app.command("run", help="Evaluate a checkpoint with Gemini and log results to Neptune")
def evaluate(
    project: Annotated[str, typer.Option("--project", "-p", help="Neptune project name")],
    run_id: Annotated[str, typer.Option("--run-id", "--neptune-run-id", "-r", help="Neptune Run ID for the checkpoint to evaluate")],
    step: Annotated[int, typer.Option("--step", "--global-step", "-s", help="Global step of the checkpoint to evaluate")],
    experiment: Annotated[str, typer.Option("--experiment", "-x", help="Experiment name for organization")],
    parent_run_id: str = typer.Option(None, "--parent-run-id", "-P", help="If run_id is forked, provide the parent run id"),
    fork_step: int = typer.Option(None, "--fork-step", "-S", help="If run_id is forked, provide the global step where the fork was made"),
    n_samples: int = typer.Option(10, "--n-samples", "-n", help="Number of samples to generate per digit"),
    gemini_model: str = typer.Option("gemma-3-27b-it", "--gemini-model", "-m", help="Gemini model to use for evaluation"),
    gemini_api_rpm: int = typer.Option(25, "--gemini-api-rpm", "-R", help="Gemini API requests per minute limit"),
):
    device = get_device()

    try:
        try:
            print(f"Loading checkpoint from {run_id=} {step=}")
            pipeline, training_info = load_checkpoint_by_run_and_step(run_id, step, device=device)
            pipeline.unet.eval()
            print(
                f"✓ Loaded checkpoint: epoch={training_info.get('epoch', 'unknown')}, step={training_info.get('step', 'unknown')}"
            )
        except Exception as e:
            if parent_run_id is None:
                raise e from e

            print(f"⚠️ Loading checkpoint from {parent_run_id=} {step=}")
            pipeline, training_info = load_checkpoint_by_run_and_step(parent_run_id, step, device=device)
            pipeline.unet.eval()
            print(
                f"✓ Loaded checkpoint: epoch={training_info.get('epoch', 'unknown')}, step={training_info.get('step', 'unknown')}"
            )
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    evaluator = GeminiEvaluator(gemini_model, gemini_api_rpm)

    print(f"Connecting to Neptune run: {run_id}")
    run = Run(
        run_id=run_id,
        resume=True,
        project=project,
        experiment_name=experiment,
        fork_run_id=parent_run_id,
        fork_step=fork_step,
        runtime_namespace="eval/runtime",
    )
    print_run_urls(run)
    log_environment(run, prefix="eval")
    run.log_configs(
        {
            "eval/config/device": device,
            "eval/config/n_samples": n_samples,
            "eval/config/gemini_model": gemini_model,
            "eval/config/gemini_api_rpm": gemini_api_rpm,
            "eval/config/global_step": step,
        }
    )

    def log_preview_scores(data: dict[str, float], progress: float):
        if progress == 1.0:  # 100% = commit metric value for the given step
            preview = False
            preview_completion = None
        else:  # progress < 100% = log preview of the metric value
            preview = True
            preview_completion = progress

        run.log_metrics(
            data=data,
            step=step,
            preview=preview,
            preview_completion=preview_completion,
        )

    evaluations = {
        "is_digit": {
            "prompt": "can you see a digit on the attached image?",
            "system_prompt": """
                you're evaluating output of a large language model, return single word 'true' or 'false'
                with no other information, no quotes, all letters lowercase
                """.replace(r"\s+", " "),
            "parse_score": lambda output, _: 1 if output.lower().strip() == "true" else 0,
        },
        "correct_digit": {
            "prompt": "what digit is it?",
            "system_prompt": """
                you're evaluating output of a large language model, return single digit between 0 and 9
                with no quotes and no other text
                """.replace(r"\s+", " "),
            "parse_score": lambda output, digit: 1 if int(output.strip()) == digit else 0,
        },
        "hand_writing": {
            "prompt": "on scale 1 (false) to 5 (true), how likely do you think this digit was written by a human?",
            "system_prompt": """
                you're evaluating output of a large language model, return single digit between 1 and 5
                with no quotes and no other text
                """.replace(r"\s+", " "),
            "parse_score": lambda output, _: int(output.strip()),
        },
    }

    all_results = {eval_name: [] for eval_name in evaluations.keys()}
    with tempfile.TemporaryDirectory() as tmpdir:
        for digit in range(10):
            print(f"\n--- Evaluating digit {digit} ---")
            digit_results = {eval_name: [] for eval_name in evaluations.keys()}

            # Generate all samples for this digit in parallel (one batch)
            print(f"Generating {n_samples} samples for digit {digit}")
            with torch.no_grad():
                result = pipeline(
                    class_labels=[digit] * n_samples,
                    batch_size=n_samples,
                    num_inference_steps=50,
                    return_pil=True,
                    show_progress=False,
                )

            for sample_idx, img in enumerate(result["images"]):
                img_file_name = f"digit{digit}_sample{sample_idx:03d}.png"
                img_path = os.path.join(tmpdir, img_file_name)
                img.save(img_path)
                print(f"Generated: {img_file_name}")

                run.log_files(
                    files={f"eval/samples/digit={digit}/sample={sample_idx:03d}.png": img_path},
                    step=step,
                )
                file = evaluator.upload_image_once(img_path)

                for eval_name, eval_config in evaluations.items():
                    run.log_string_series(
                        data={
                            f"eval/{eval_name}/prompt": eval_config["prompt"],
                            f"eval/{eval_name}/system_prompt": eval_config["system_prompt"],
                        },
                        step=step,
                    )

                    eval_output = evaluator.evaluate(file, eval_config["prompt"], eval_config["system_prompt"])
                    score = eval_config["parse_score"](eval_output, digit)
                    print(f"  {eval_name}: {eval_output} -> {score}")

                    digit_results[eval_name].append(score)
                    all_results[eval_name].append(score)

                    run.log_metrics(
                        data={f"eval/{eval_name}/scores/digit={digit}/sample={sample_idx:03d}": score},
                        step=step,
                    )
                    log_preview_scores(
                        data={
                            f"eval/{eval_name}/scores/digit={digit}/avg": statistics.mean(digit_results[eval_name]),
                            f"eval/{eval_name}/scores/digit={digit}/max": max(digit_results[eval_name]),
                            f"eval/{eval_name}/scores/digit={digit}/min": min(digit_results[eval_name]),
                        },
                        progress=(sample_idx + 1) / n_samples,
                    )
                    log_preview_scores(
                        data={
                            f"eval/{eval_name}/scores/avg": statistics.mean(all_results[eval_name]),
                            f"eval/{eval_name}/scores/max": max(all_results[eval_name]),
                            f"eval/{eval_name}/scores/min": min(all_results[eval_name]),
                        },
                        progress=len(all_results[eval_name]) / (n_samples * 10),
                    )

    # Wait for Neptune processing and close
    run.wait_for_processing(timeout=60)
    run.close()
    print("")
    print("✅ Evaluation completed!")
    print(f"Neptune run URL: {run.get_run_url()}")


if __name__ == "__main__":
    app()
