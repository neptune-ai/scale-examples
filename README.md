<div align="center">
    <img src="https://neptune.ai/wp-content/uploads/2024/07/Facebook-cover-image.png" width="1500" />
    &nbsp;
 <h1><a href="https://neptune.ai">neptune.ai</a> Scale examples</h1>
</div>

## What is neptune.ai?

Neptune is the most scalable experiment tracker for teams that train foundation models.  

Log millions of runs, view and compare them all in seconds. Effortlessly monitor and visualize months-long model training with multiple steps and branches.  

Deploy Neptune on your infra from day one, track 100% of your metadata and get to the next big AI breakthrough faster.
&nbsp;

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
