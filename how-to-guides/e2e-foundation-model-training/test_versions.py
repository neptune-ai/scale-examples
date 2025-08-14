from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker

pipe = Pipeline(
    tasks="leaderboard|truthfulqa:mc|0|0",
    pipeline_parameters=PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=2,
        dataset_loading_processes=1),
    model_config=TransformersModelConfig(
        model_name="gpt2", 
        dtype="float16", 
        batch_size=1, 
        use_chat_template=False,
        device="cpu"),
    evaluation_tracker=EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        hub_results_org="your user name",
    )
)
pipe.evaluate()
pipe.show_results()

results = pipe.get_results() # this is a dictionary of the results
print(results["results"])
#print(pipe.evaluation_tracker.metrics_logger.aggregate())