<div align="center">
 <img src="https://raw.githubusercontent.com/neptune-ai/neptune-client/assets/readme/Github-cover-022025.png" width="1500" />
 &nbsp;
 <h1><a href="https://neptune.ai">neptune.ai</a> examples</h1>
</div>

## What is neptune.ai?

Neptune is an experiment tracker purpose-built for foundation model training.<br>
<br>
With Neptune, you can monitor thousands of per-layer metrics—losses, gradients, and activations—at any scale. Visualize them with no lag and no missed spikes. Drill down into logs and debug training issues fast. Keep your model training stable while reducing wasted GPU cycles.<br>

## 📚Examples

In this repo, you'll find tutorials and examples of using Neptune Scale.

> [!NOTE]
> These examples only work with the [`neptune-scale`](https://github.com/neptune-ai/neptune-client-scale) Python client, which is in beta.
>
> You can't use these with the stable Neptune `2.x` versions currently available to SaaS and self-hosting customers. For examples corresponding to Neptune `2.x`, see https://github.com/neptune-ai/examples.

## 🎓How-to guides

### 👶 First steps

| | Docs | Neptune
| ----------- | :---: | :---:
| Quickstart | [![docs]](https://docs-beta.neptune.ai/quickstart) |
| Track and organize runs | [![docs]](https://docs-beta.neptune.ai/experiments_table) | [![neptune]](https://scale.neptune.ai/o/neptune/org/LLM-training-example/runs/table?viewId=9d0e03d5-d0e9-4c0a-a546-f065181de1d2&detailsTab=metadata&dash=table&type=run&compare=auto-5)

### 🧑 Deeper dive

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Resume run or other object | [![docs]](https://docs-beta.neptune.ai/resume_run)
| Use Neptune in HPO jobs | [![docs]](https://docs-beta.neptune.ai/hpo_tutorial) | [![neptune]](https://scale.neptune.ai/o/examples/org/hpo/runs/table?viewId=9d44261f-32a1-42e7-96ff-9b35edc4be66) | [![github]](how-to-guides/hpo/notebooks/Neptune_HPO.ipynb) | [![colab]](https://colab.research.google.com/github/neptune-ai/scale-examples/blob/master/how-to-guides/hpo/notebooks/Neptune_HPO.ipynb) |

### 👨 Advanced concepts

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| DDP training scripts | | | [![github]](how-to-guides/ddp-training/scripts/) | |
## 🛠️ Other utilities

### 🧳 Migration tools

| | GitHub
| - | :-:
| Import runs from Weights & Biases | [![github]](utils/migration_tools/from_wandb/)


## 🔍 Cannot find what you are looking for?
Check out our [docs](https://docs-beta.neptune.ai/), our [blog](https://neptune.ai/blog), or reach out to us at support@neptune.ai.


<!--- Resources -->
[docs]: https://neptune.ai/wp-content/uploads/2023/06/file_icon.svg "Read the documentation"
[neptune]: https://neptune.ai/wp-content/uploads/2023/01/Signet-svg-16x16-1.svg "See Neptune example"
[github]: https://neptune.ai/wp-content/uploads/2023/06/Github-Monochrome-1.svg "See code on GitHub"
[colab]: https://neptune.ai/wp-content/uploads/colab-icon.png "Open in Colab"
## Examples

This repo contains tutorials and examples of how to use Neptune.

|                                 | Docs                         | Neptune                                 | GitHub                         | Colab                      |
| ------------------------------- | ---------------------------- | --------------------------------------- | ------------------------------ | -------------------------- |
| Quickstart                      | [![docs-icon]][quickstart]   | [![neptune-icon]][quickstart-example]   | [![github-icon]][qs-notebook]  | [![colab-icon]][qs-colab]  |
| Log different types of metadata | [![docs-icon]][log-metadata] | [![neptune-icon]][log-metadata-example] |                                |                            |
| Organize and filter runs        | [![docs-icon]][runs-table]   | [![neptune-icon]][runs-table-example]   |                                |                            |
| Resume run or other object      | [![docs-icon]][resume-run]   |                                         |                                |                            |
| Use Neptune in HPO jobs         | [![docs-icon]][hpo]          | [![neptune-icon]][hpo-example]          | [![github-icon]][hpo-notebook] | [![colab-icon]][hpo-colab] |
| Debug training runs             | [![docs-icon]][debug]        | [![neptune-icon]][debug-example]        | [![github-icon]][debug-notebook] | [![colab-icon]][debug-colab] |

### Integrations
| | Docs | Neptune | GitHub | Colab |
| -- | :--: | :--: | :--: | :--: |
|  PyTorch Lightning  | [![docs-icon]][lightning] | [![neptune-icon]][lightning-example] | [![github-icon]][lightning-notebook] | [![colab-icon]][lightning-colab] |

### Migration tools

|                                         |            Docs             |               GitHub               |
| --------------------------------------- | :-------------------------: | :--------------------------------: |
| Import runs from Weights & Biases       | [![docs-icon]][from-wandb]  | [![github-icon]][from-wandb-code]  |
| Migrate code from Neptune Legacy client | [![docs-icon]][from-legacy] | [![github-icon]][from-legacy-code] |

### Monitoring tools

|                         |                  Docs                   |                      Neptune                       |                     GitHub                     |
| ----------------------- | :-------------------------------------: | :------------------------------------------------: | :--------------------------------------------: |
| Hardware monitoring     |   [![docs-icon]][hardware-monitoring]   |   [![neptune-icon]][hardware-monitoring-example]   |   [![github-icon]][hardware-monitoring-code]   |
| PyTorch model internals | [![docs-icon]][pytorch-model-internals] | [![neptune-icon]][pytorch-model-internals-example] | [![github-icon]][pytorch-model-internals-code] |

### Visualization tools

|                           |                 GitHub                 |
| ------------------------- | :------------------------------------: |
| Compare media file-series | [![github-icon]][file-comparison-code] |

## Can't find what you're looking for?

- [Visit the documentation &rarr;][docs]
- [Check out the blog &rarr;][blog]
- Visit our [Support Center](https://support.neptune.ai/) to get help or contact the Support team.

<!-- Internal -->

[from-wandb-code]: utils/migration_tools/from_wandb/
[from-legacy-code]: utils/migration_tools/from_legacy_neptune/
[hpo-notebook]: how-to-guides/hpo/notebooks/Neptune_HPO.ipynb
[hpo-colab]: https://colab.research.google.com/github/neptune-ai/scale-examples/blob/master/how-to-guides/hpo/notebooks/Neptune_HPO.ipynb
[pytorch-model-internals-code]: utils/monitoring_tools/pytorch_model_internals/
[file-comparison-code]: utils/visualization_tools/file_comparison_app/
[qs-notebook]: how-to-guides/quickstart/notebooks/neptune_quickstart.ipynb
[qs-colab]: https://colab.research.google.com/github/neptune-ai/scale-examples/blob/master/how-to-guides/quickstart/notebooks/neptune_quickstart.ipynb
[lightning-notebook]: TODO: final notebook URL in GH
[lightning-colab]: TODO: final colab URL
[debug-notebook]: how-to-guides/debug-model-training-runs/debug_training_runs.ipynb
[debug-colab]: https://colab.research.google.com/github/neptune-ai/scale-examples/blob/master/how-to-guides/debug-model-training-runs/notebooks/debug_training_runs.ipynb

<!-- External -->

[blog]: https://neptune.ai/blog
[docs]: https://docs.neptune.ai/
[from-legacy]: https://docs.neptune.ai/migration_neptune
[from-wandb]: https://docs.neptune.ai/migration_wandb
[hardware-monitoring]: https://docs.neptune.ai/hardware_monitoring
[hardware-monitoring-code]: utils/monitoring_tools/hardware_monitoring/
[hardware-monitoring-example]: https://scale.neptune.ai/o/examples/org/showcase/runs/details?viewId=9f113328-75aa-4c61-9aa8-5bbdffa90879&detailsTab=dashboard&dashboardId=9f11330c-e4ff-413a-9faa-9e10e5b3f7ee&runIdentificationKey=hardware_monitoring&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&suggestionsEnabled=false&query=&experimentOnly=true
[hpo]: https://docs.neptune.ai/hpo_tutorial
[hpo-example]: https://scale.neptune.ai/o/examples/org/hpo/runs/table?viewId=9d44261f-32a1-42e7-96ff-9b35edc4be66
[log-metadata]: https://docs.neptune.ai/log_metadata
[log-metadata-example]: https://scale.neptune.ai/o/examples/org/LLM-Pretraining/runs/details?viewId=9e6a41f4-69a5-4d9f-951c-b1304f2acf12&detailsTab=dashboard&dashboardId=9e6a5c4c-0c39-491f-9811-87eeb39a2603&runIdentificationKey=LLM-29&type=run&compare=uMlyIDUTmecveIHVma0eEB95Ei5xu8F_9qHOh0nynbtM
[pytorch-model-internals]: https://docs.neptune.ai/utility_scripts/torchwatcher
[pytorch-model-internals-example]: https://scale.neptune.ai/examples/showcase/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=9f67bd03-4080-4d47-83b2-36836b03351c&runIdentificationKey=torch-watcher-example&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&experimentOnly=true
[quickstart]: https://docs.neptune.ai/quickstart
[quickstart-example]: https://scale.neptune.ai/examples/quickstart/reports/9ea26258-2aed-4441-9b58-bab619215f6f
[resume-run]: https://docs.neptune.ai/resume_run
[runs-table]: https://docs.neptune.ai/runs_table
[runs-table-example]: https://scale.neptune.ai/o/examples/org/LLM-Pretraining/runs/table?viewId=9e746462-f045-4ff2-9ac4-e41fa349b04d&detailsTab=dashboard&dash=table&type=run&compare=auto-5
[lightning]: https://docs.neptune.ai/integrations/pytorch_lightning
[lightning-example]: https://scale.neptune.ai/o/examples/org/pytorch-lightning/runs/table?viewId=9ea6121c-42a7-4ece-83b2-c591044837e7
[debug]: https://docs.neptune.ai/debug_runs_tutorial
[debug-example]: https://scale.neptune.ai/o/examples/org/debug-training-metrics/runs/table?viewId=standard-view&dash=table&compareChartsFilter-compound=udzSoRe3VmlvolZ8TbuB_zvfcAcgJmla8UuNku1rGWdg

<!-- Clickable icons -->

[docs-icon]: https://neptune.ai/wp-content/uploads/2023/06/file_icon.svg "Read the documentation"
[neptune-icon]: https://neptune.ai/wp-content/uploads/2023/01/Signet-svg-16x16-1.svg "See Neptune example"
[github-icon]: https://neptune.ai/wp-content/uploads/2023/06/Github-Monochrome-1.svg "See code on GitHub"
[colab-icon]: https://neptune.ai/wp-content/uploads/colab-icon.png "Open in Colab"
