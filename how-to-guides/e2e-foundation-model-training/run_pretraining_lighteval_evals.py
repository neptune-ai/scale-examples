import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
# from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from lighteval.models.transformers.transformers_model import TransformersModelConfig

import warnings
warnings.filterwarnings('ignore')

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main():
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        hub_results_org="your user name",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        # env_config=EnvConfig(cache_dir="tmp/"),
        custom_tasks_directory=None, # if using a custom task
        # Remove the 2 parameters below once your configuration is tested
        # override_batch_size=1,
        max_samples=10
    )

    model_config = TransformersModelConfig(
        model_name="./results/checkpoint-20", # TODO: Change to the model you want to evaluate
        dtype="float16",
        use_chat_template=False,
        batch_size=1,
    )


    # model = GPT2LMHeadModel.from_pretrained("./results/checkpoint-20")

    # task = "lighteval|math|0|1"
    task = "lighteval|truthfulqa:gen|0|1"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        # model=model
    )

    pipeline.evaluate()
    # pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()