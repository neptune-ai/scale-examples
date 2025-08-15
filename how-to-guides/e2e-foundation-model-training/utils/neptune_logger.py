from transformers import TrainerCallback

def _prefix_dict(dict, prefix):
    return {f"{prefix}/{k}": v for k, v in dict.items()}

class NeptuneCallback(TrainerCallback):
        def __init__(self, run, watcher=None):
            self.run = run
            self.watcher = watcher

        def on_init_end(self, args, state, control, **kwargs):
            self.run.log_configs(
                _prefix_dict(args.to_dict(), "config"), 
                flatten=True, 
                cast_unsupported=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and state.is_local_process_zero:
                if not control.should_evaluate:
                    metrics = {f"train/{k}": v for k, v in logs.items()}
                    self.run.log_metrics(
                        data={**metrics},
                        step=state.global_step
                    )
                    if self.watcher is not None:
                        self.watcher.watch(
                            step=state.global_step,
                            track_gradients=True,
                            track_parameters=False,
                            track_activations=True,
                        )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and state.is_local_process_zero:
                metrics = {f"eval/{k}": v for k, v in metrics.items()}
                self.run.log_metrics(
                    data={**metrics},
                    step=state.global_step
                )