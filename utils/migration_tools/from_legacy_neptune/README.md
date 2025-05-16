# Copying runs from Neptune Legacy (2.x) to Neptune (3.x)

This script helps you copy runs from Neptune Legacy (2.x) to Neptune (3.x).

---
## Changelog
- 2025-05-16 - Initial release

## Quick Start

Install the required packages:
```bash
pip install -U neptune-scale neptune
```

Run the migration script:
```bash
python runs_migrator.py \
  --legacy-token <NEPTUNE_2.X_API_TOKEN> \
  --new-token <NEPTUNE_3.X_API_TOKEN> \
  --new-workspace <NEPTUNE_3.X_WORKSPACE_NAME> \
  --legacy-project <LEGACY_WORKSPACE/PROJECT_NAME> \
```

---

## Arguments

- `--legacy-token` (required): API token for the legacy Neptune workspace (2.x).
- `--new-token` (required): API token for the new Neptune workspace (3.x).
- `--new-workspace` (required): Name of the new workspace in Neptune 3.x.
- `--legacy-project` (required): Name of the legacy project in the format `WORKSPACE_NAME/PROJECT_NAME`.
- `--query` (optional): Query filter for runs to be copied ([NQL syntax](https://docs-legacy.neptune.ai/usage/nql/)).
- `--max-workers` (optional): Maximum number of parallel workers to use for copying runs. Defaults to using ThreadPoolExecutor's default.

---

## Prerequisites
- A Neptune 2.x workspace with runs to migrate.
- A Neptune 3.x workspace with write access to migrate the runs to.
- Latest versions of the `neptune-scale` and `neptune` Python packages installed.

---

## How It Works

- Runs are copied from the legacy project to a project with the same name, key and visibility in the new workspace.
- The script fetches all runs from the specified legacy project that match the query.
- For each run, it copies supported metadata and metrics to a new run in the target workspace/project.
- **Namespace mapping:**
  - The `monitoring/` namespace in the source run is copied to the `runtime/` namespace in the target run.
  - All `sys` fields (except a few) are copied to the `legacy_sys` namespace.
- Unsupported metadata (see Caveats) is skipped.
- Temporary files are stored in a `.tmp_*` directory for troubleshooting. You can delete this folder after verifying the migration.

---

## Caveats and Limitations
- Runs can be copied only one project at a time.
- If a project with the same key as the source project already exists in the new workspace, the script will silently fail.
- Avoid creating new runs in the source project while the script is running as these might not be copied.
- Timestamp values appended to each step of series metrics are in the local timezone of the script execution environment. This can lead to variations between the source and target run charts if the X-axis is set to relative time and the source run was created in a different timezone.

### The following metadata are not copied at all
- Project description and members
- Project and model metadata
- Artifacts†
- FileSet (including source code)†
- All FileSeries objects†
  - Once FileSeries are supported in Neptune 3.x, the custom steps and descriptions of files in FileSeries will still not be copied
- Attributes:
  - Run state: `sys/state`
  - Git info: `source_code/git`

† Support for these will be added in future releases.

### The following metadata are not copied in the same format
- The `monitoring` namespace in the source run is copied to the `runtime` namespace in the target run
- All `sys` fields except `state`, `description`, `name`, `custom_run_id`, `tags`, and `group_tags` are copied to the `legacy_sys` namespace
- If the source run does not have a `custom_run_id`, the source run id (`sys/id`) will be used instead as the `run_id` of the target run
- `sys/running_time` and `sys/monitoring_time` are copied to `legacy_sys/running_time` and `legacy_sys/monitoring_time` respectively as floats in seconds
- `sys/size` is copied to `legacy_sys/size` as an integer in bytes

---

## Instructions

1. Execute `runs_migrator.py` with the required arguments (see Quick Start).
1. The script will generate run logs in the working directory. You can modify this location by editing the `setup_logging()` function.

---

## Troubleshooting

- **Project not found:** Ensure the `--legacy-project` is in the correct format and exists in your legacy workspace.
- **Authentication errors:** Double-check your API tokens and permissions in both the source and target projects.
- **Partial migration:** Check the log file for errors and rerun the script as needed.
- **Performance:** Increase `--max-workers` for faster migration, but be mindful of API rate limits.

---

## FAQ

**Q: Can I rerun the script if it fails?**  
A: Yes, the script is idempotent for already-migrated runs.

**Q: Are artifacts and files copied?**  
A: Not yet. See the Caveats section for details.

**Q: How do I filter which runs to migrate?**  
A: Use the `--query` argument with [NQL syntax](https://docs-legacy.neptune.ai/usage/nql/).

---

## Documentation

- [Neptune 2.x Documentation](https://docs-legacy.neptune.ai/)
- [Neptune 3.x Documentation](https://docs.neptune.ai/)
- [NQL Query Syntax](https://docs-legacy.neptune.ai/usage/nql/)

---

## Post-Migration Checklist

[ ] Review the logs for any errors. They also contain the URLs of both the source and target runs.  
[ ] Add users to the new project and assign them relevant roles  
[ ] Add project description  
[ ] Recreate saved views, dashboards, and reports in the new project  
[ ] Delete the `.tmp_{PROJECT_NAME}_migration_%Y%m%d%H%M%S` folder in the working directory after verifying the migration  

---

## Support and Feedback

We welcome your feedback and contributions to help improve the script. Please submit any issues or feature requests as [GitHub Issues](https://github.com/neptune-ai/scale-examples/issues).

For help, reach out via our support channels:
- [Support portal](https://support.neptune.ai)
- [Email](mailto:support@neptune.ai)
- In-app chat

---

## License

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
