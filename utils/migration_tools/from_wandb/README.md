# Migrating from W&B to Neptune

This script allows you to copy run metadata from W&B to Neptune.

## Changelog

- **v0.3.0** (2025-07-24)
  - Updated script to use new `flatten` and `cast_unsupported` parameters in `log_configs()`. Collections like lists, sets, and tuples are now cast to strings rather than being expanded into multiple keys.

- **v0.2.0** (2025-06-16)
  - Added console logs and file support.
  - Updated namespace of hardware metrics to `runtime` from `system`.
  - Updated script to use only CLI arguments.
  - Defaulted `num-workers` to 1 instead of `ThreadPoolExecutor`'s defaults.
  - Replaced `.` in system metric names with `/` to match the Neptune metric namespace format.

- **v0.1.0** (2025-01-08)
  - Initial release

## Prerequisites
- A Weights and Biases account, `wandb` library installed, and environment variables set.
- A Neptune account, the latest `neptune-scale` Python library installed, and environment variables set. For details, see the [docs][docs-setup].

> [!NOTE]
> The script has been tested with `wandb==0.20.1` and `neptune-scale==0.18.0`.

## Usage

Run the script with CLI arguments:

```sh
python wandb_to_neptune.py \
  [--wandb-entity <wandb_entity>] \
  [--neptune-workspace <neptune_workspace>] \
  [--projects <project1,project2,...>] \
  [--num-workers <int>] \
  [--log-level <DEBUG|INFO|WARNING|ERROR|CRITICAL>] \
  [--timeout <int>]
```

### Arguments
- `--wandb-entity` (str, optional): W&B entity name. Defaults to your W&B default entity if available.
- `--neptune-workspace` (str, optional): Neptune workspace name. Defaults to the first part of the [`NEPTUNE_PROJECT`][docs-neptune-project-env-variable] environment variable.
- `--projects` (str, optional): Comma-separated list of W&B projects to copy. If omitted or blank, **all projects** in the entity will be copied.
- `--num-workers` (int, optional): Number of worker threads to use. Defaults to 1. This number is recommended for stability.  
- `--log-level` (str, optional): Logging verbosity. One of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `INFO`.  
- `--timeout` (int, optional): Timeout in seconds for each run/project. Defaults to 600.

### Example
Copy all projects from your default W&B entity to your default Neptune workspace:
```sh
python wandb_to_neptune.py
```

Copy specific projects with debug logging, 2 worker threads, and a 5 minute timeout:
```sh
python wandb_to_neptune.py --wandb-entity myentity --neptune-workspace myworkspace --projects proj1,proj2 --log-level DEBUG --num-workers 2 --timeout 300
```

### Note
- If they don't exist yet, Neptune projects corresponding to the W&B projects will be created as [*private*][docs-project-access] with project description set to *Exported from <W&B project URL>*. To change the project visibility or description, use the web app or update the `create_project()` function in `copy_project()`.  
- A `tmp_<timestamp>` directory is created in the current working directory to store the files. Ensure that the current working directory is writable, and that there is enough disk space. You can delete this directory after the script has finished running and the required sanity checks have been performed.

## Metadata mapping from W&B to Neptune

| Metadata | W&B | Neptune |
| :-: | :-: | :-: |
| Project name | example_project | example-project<sup>1</sup> |
| Project URL | project.url | Project description |
| Run name | run.name | run.sys.name |
| Run ID | run.id | run.sys.custom_run_id<sup>2</sup> |
| Notes | run.notes | run.sys.description |
| Tags | run.tags | run.sys.tags |
| Group | run.group | run.sys.group_tags |
| Config | run.config | run.config<sup>3</sup> |
| Run summary | run.summary | run.summary<sup>3</sup> |
| Run metrics | run.scan_history() | run.<METRIC_NAME><sup>4</sup> |
| System metrics | run.history(stream="system") | run.runtime.<METRIC_NAME><sup>5</sup> |
| Console logs | Files > `output.log` | run.runtime.stdout<sup>6</sup> |
| Source code | Files > `code/` | run.source_code.files<sup>7</sup> |
| Requirements | Files > `requirements.txt` | run.source_code.requirements |
| Checkpoints | Files > `ckpt/`/`checkpoint/` | run.checkpoints |
| Other files | Files | run.files |
| All W&B attributes | run.* | run.wandb.* |

<sup>1</sup> Underscores `_` in a W&B project name are replaced by a hyphen `-` in Neptune  
<sup>2</sup> Passing the wandb.run.id as neptune.run.custom_run_id ensures that duplicate Neptune runs are not created for the same W&B run even if the script is run multiple times  
<sup>3</sup> Values are converted to a string in Neptune  
<sup>4</sup> `_step` and `_timestamp` associated with a metric are logged as `step` and `timestamp` respectively with a Neptune metric  
<sup>5</sup> `system.` prefix is removed when logging to Neptune. The `.` in the metric name is replaced with `/` to match the Neptune metric namespace format.  
<sup>6</sup> Lines from the `output.log` file in W&B are logged as `stdout` in Neptune. Not all W&B runs have an `output.log` file. Even when they do, the file doesn't necessarily reflect the information from  the console logs displayed in the _Logs_ section of the W&B run. Steps are logged as the line number of the `output.log` file, and the timestamp associated with each step is the current time of the script execution, not the timestamp logged to the W&B run.  
<sup>7</sup> All source code files are rendered as plain text in the Neptune web app.

## What is not exported
- Project-level metadata
- Models
- W&B specific objects and data types
- Forking details
- `run.summary` keys starting with `_`*
- Metrics and W&B attributes starting with `_`*

\* These have been excluded at the code level to prevent redundancy and noise, but can be included.

## Post-migration
* W&B Workspace views can be recreated using Neptune's [overlaid charts][docs-charts] and [reports][docs-reports]
* W&B Runs table views can be recreated using Neptune's [custom views][docs-custom-views]
  ![Example W&B Runs table view recreated in Neptune](https://neptune.ai/wp-content/uploads/2025/01/WB_NeptuneScale.png)
* W&B Run Overview can be recreated using Neptune's [custom dashboards][docs-custom-dashboards]

## Troubleshooting

- If you see timeouts, try increasing the `--timeout` value.
- If the migration is slow, you can increase `--num-workers`. Note that high concurrency may cause issues with the W&B and Neptune clients and hit API rate limits.
- If you hit API rate limits, try reducing the number of workers or spacing out your migrations.
- For detailed error messages, check the log file (`wandb_to_neptune_<timestamp>.log`).  For more detailed logs, reduce the `--log-level` to `DEBUG`.  
- If any projects fail to copy, the script will exit with a non-zero code.

## Support and feedback

We welcome your feedback and contributions to help improve the script. Please submit any issues or feature requests as [GitHub Issues](https://github.com/neptune-ai/scale-examples/issues)

## License

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


[docs-charts]: https://docs.neptune.ai/charts/
[docs-custom-dashboards]: https://docs.neptune.ai/custom_dashboard/
[docs-custom-views]: https://docs.neptune.ai/runs_table#custom-views
[docs-project-access]: https://docs.neptune.ai/project_access
[docs-reports]: https://docs.neptune.ai/reports/
[docs-setup]: https://docs.neptune.ai/setup
[docs-neptune-project-env-variable]: https://docs.neptune.ai/environment_variables/authentication/#neptune_project
