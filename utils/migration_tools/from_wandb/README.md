# Migrating from W&B to Neptune Scale

This script allows you to copy run metadata from W&B to Neptune Scale.

## Prerequisites
- A Weights and Biases account, `wandb` library installed, and environment variables set.
- A Neptune Scale account, `neptune-scale` python library installed, and environment variables set. For details, see the [docs][docs-setup].

## Instructions

To use the script, follow these steps:

1. Run `wandb_to_neptune.py`.
1. Enter the source W&B entity name. Leave blank to use your default entity.
1. Enter the destination Neptune workspace name. Leave blank to read from the `NEPTUNE_PROJECT` environment variable.
1. Enter the number of workers to use to copy the metadata. Leave blank to select the   number of workers automatically.
1. Enter the W&B projects you want to export as comma-separated values. Leave blank to export all projects.
1. The script will generate run logs in the working directory. You can change the directory with `logging.basicConfig()`. Live progress bars will also be rendered in the console.
1. Neptune Scale projects corresponding to the W&B projects will be created with [*private*][docs-project-access] visibility if they don't exist. You can change the visibility later from the WebApp once the project has been created, or by updating the `create_project()` function in `copy_project()`.
1. The project description will be set as *Exported from <W&B project URL>*. You can change the description later from the WebApp once the project has been created, or by updating the `create_project()` function in `copy_project()`.

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
| System metrics | run.history(stream="system") | run.system.<METRIC_NAME><sup>5</sup> |
| All W&B attributes | run.* | run.wandb.* |

<sup>1</sup> Underscores `_` in a W&B project name are replaced by a hyphen `-` in Neptune  
<sup>2</sup> Passing the wandb.run.id as neptune.run.custom_run_id ensures that duplicate Neptune runs are not created for the same W&B run even if the script is run multiple times  
<sup>3</sup> Values are converted to a string in Neptune  
<sup>4</sup> `_step` and `_timestamp` associated with a metric are logged as `step` and `timestamp` respectively with a Neptune metric  
<sup>5</sup> `system.` prefix is removed when logging to Neptune

## What is not exported
- Project-level metadata
- All files (artifacts, source code, requirements.txt, images, etc.)
- Models
- W&B specific objects and data types
- Forking details
- `run.summary` keys starting with `_`†
- Metrics and W&B attributes starting with `_`†

† These have been excluded at the code level to prevent redundancy and noise, but can be included.

## Post-migration
* W&B Workspace views can be recreated using Neptune's [overlaid charts][docs-charts] and [reports][docs-reports]
* W&B Runs table views can be recreated using Neptune's [custom views][docs-custom-views]
  ![Example W&B Runs table view recreated in Neptune](https://neptune.ai/wp-content/uploads/2025/01/WB_NeptuneScale.png)
* W&B Run Overview can be recreated using Neptune's [custom dashboards][docs-custom-dashboards]

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