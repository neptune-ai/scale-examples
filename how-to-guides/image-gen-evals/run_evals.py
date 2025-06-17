import os
import time
import tempfile
import torch
import argparse
import statistics
from checkpoint_utils import load_checkpoint_by_run_and_step
from net import get_device
from google import genai
from google.genai.types import File
from neptune_scale import Run


parser = argparse.ArgumentParser()
parser.add_argument(
    "--neptune-project", type=str, default="neptune/class-conditioned-difussion"
)
parser.add_argument("--neptune-experiment", type=str, default="debug-001")
parser.add_argument(
    "--neptune-run-id",
    type=str,
    required=True,
    help="Neptune run ID for the checkpoint to evaluate",
)
parser.add_argument(
    "--global-step",
    type=int,
    required=True,
    help="Global step of the checkpoint to evaluate",
)
parser.add_argument(
    "--n-samples", type=int, default=10, help="Number of samples to generate per digit"
)
parser.add_argument(
    "--gemini-model",
    type=str,
    default="gemma-3-27b-it",
    help="Gemini model to use for evaluation",
)
parser.add_argument(
    "--gemini-api-rpm",
    type=int,
    default=20,
    help="Gemini API requests per minute limit",
)


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


def main():
    args = parser.parse_args()
    device = get_device()

    # Load checkpoint
    print(f"Loading checkpoint from run {args.neptune_run_id}, step {args.global_step}")
    try:
        pipeline, training_info = load_checkpoint_by_run_and_step(
            args.neptune_run_id, args.global_step, device=device
        )
        pipeline.unet.eval()
        print(
            f"✓ Loaded checkpoint: epoch={training_info.get('epoch', 'unknown')}, step={training_info.get('step', 'unknown')}"
        )
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    evaluator = GeminiEvaluator(args.gemini_model, args.gemini_api_rpm)

    # Connect to Neptune run for logging
    print(f"Connecting to Neptune run: {args.neptune_run_id}")
    run = Run(run_id=args.neptune_run_id, resume=True, project=args.neptune_project)

    # Log evaluation configuration
    run.log_configs(
        {
            "evals/config/device": device,
            "evals/config/n_samples": args.n_samples,
            "evals/config/gemini_model": args.gemini_model,
            "evals/config/gemini_api_rpm": args.gemini_api_rpm,
            "evals/config/global_step": args.global_step,
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
            step=args.global_step,
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
            "parse_score": lambda output, _: 1
            if output.lower().strip() == "true"
            else 0,
        },
        "correct_digit": {
            "prompt": "what digit is it?",
            "system_prompt": """
                you're evaluating output of a large language model, return single digit between 0 and 9
                with no quotes and no other text
                """.replace(r"\s+", " "),
            "parse_score": lambda output, digit: 1
            if int(output.strip()) == digit
            else 0,
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
            print(f"Generating {args.n_samples} samples for digit {digit}")
            with torch.no_grad():
                result = pipeline(
                    class_labels=[digit] * args.n_samples,
                    batch_size=args.n_samples,
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
                    files={
                        f"evals/samples/digit={digit}/sample={sample_idx:03d}.png": img_path
                    },
                    step=args.global_step,
                )
                file = evaluator.upload_image_once(img_path)

                for eval_name, eval_config in evaluations.items():
                    run.log_string_series(
                        data={
                            f"evals/{eval_name}/prompt": eval_config["prompt"],
                            f"evals/{eval_name}/system_prompt": eval_config[
                                "system_prompt"
                            ],
                        },
                        step=args.global_step,
                    )

                    eval_output = evaluator.evaluate(
                        file, eval_config["prompt"], eval_config["system_prompt"]
                    )
                    score = eval_config["parse_score"](eval_output, digit)
                    print(f"  {eval_name}: {eval_output} -> {score}")

                    digit_results[eval_name].append(score)
                    all_results[eval_name].append(score)

                    run.log_metrics(
                        data={
                            f"evals/{eval_name}/scores/digit={digit}/sample={sample_idx:03d}": score
                        },
                        step=args.global_step,
                    )
                    log_preview_scores(
                        data={
                            f"evals/{eval_name}/scores/digit={digit}/avg": statistics.mean(
                                digit_results[eval_name]
                            ),
                            f"evals/{eval_name}/scores/digit={digit}/max": max(
                                digit_results[eval_name]
                            ),
                            f"evals/{eval_name}/scores/digit={digit}/min": min(
                                digit_results[eval_name]
                            ),
                        },
                        progress=(sample_idx + 1) / args.n_samples,
                    )
                    log_preview_scores(
                        data={
                            f"evals/{eval_name}/scores/avg": statistics.mean(
                                all_results[eval_name]
                            ),
                            f"evals/{eval_name}/scores/max": max(
                                all_results[eval_name]
                            ),
                            f"evals/{eval_name}/scores/min": min(
                                all_results[eval_name]
                            ),
                        },
                        progress=len(all_results[eval_name])
                        / (args.n_samples * 10),
                    )

    # Wait for Neptune processing and close
    run.wait_for_processing(timeout=60)
    run.close()
    print("")
    print("✅ Evaluation completed!")
    print(f"Neptune run URL: {run.get_run_url()}")


if __name__ == "__main__":
    main()
