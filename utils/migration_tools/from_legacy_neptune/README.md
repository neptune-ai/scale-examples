# Copying runs from Neptune Legacy (2.x) to Neptune (3.x)

This script helps you copy runs from Neptune Legacy (2.x) to Neptune (3.x).

---
## Changelog
- 2025-05-19 - Initial release

---

## Prerequisites
- A workspace in Neptune 3.x, and an API token that has access to the workspace.
- Latest versions of the `neptune-scale` and `neptune` Python packages installed:
  ```bash
  pip install -U neptune-scale neptune
  ```

---

## Quickstart
```bash
python runs_migrator.py \
  --legacy-token <NEPTUNE_2.X_API_TOKEN> \
  --new-token <NEPTUNE_3.X_API_TOKEN> \
  --new-workspace <NEPTUNE_3.X_WORKSPACE_NAME> \
  --legacy-project <LEGACY_WORKSPACE/PROJECT_NAME> \
```

---

## Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--legacy-token` | Yes | API token for the legacy Neptune workspace (2.x). |
| `--new-token` | Yes | API token for the new Neptune workspace (3.x). |
| `--legacy-project` | Yes | Name of the legacy project in the format `WORKSPACE_NAME/PROJECT_NAME`. |
| `-w`, `--new-workspace` | Yes | Name of the new workspace in Neptune 3.x. |
| `--new-project` | No | Name of the new project in the format `PROJECT_NAME`. The project will be created if it doesn't already exist. If not provided, the project name will be the same as the legacy project name. |
| `--max-workers` | No | Maximum number of parallel workers to use for copying runs. Defaults to ThreadPoolExecutor’s default. See [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) for details. |

---

## How it works
- The target project is created in the new workspace if it doesn't already exist.
  - Metrics and parameters are copied from memory
    - The `monitoring/` namespace in the source run is copied to the `runtime/` namespace in the target run.
    - Most `sys` attributes are copied to the `legacy_sys` namespace. For details, see [Notes and Limitations](#notes-and-limitations).
  - Files are first downloaded to a temporary directory and then uploaded to the new run.
    - Temporary files are stored in a `.tmp_*` directory for troubleshooting. You can delete this folder after verifying the migration. Ensure that you have sufficient space to store the temporary files.
  - Unsupported metadata (see Notes and Limitations) is skipped.

---

## Notes and limitations
- Runs can be copied only from one project at a time.
- If a new project name isn't provided and a project with the same key as the source project already exists in the new workspace, the script will silently fail.  
- Avoid creating new runs in the source project while the script is running as these might not be copied.
- Timestamp values appended to each step of series metrics are in the local timezone of the script execution environment. This can lead to variations between the source and target run charts if the X-axis is set to relative time and the source run was created in a different timezone.

### Data that isn't copied in the same format
- If the source run doesn't have a `custom_run_id`, the source run ID (`sys/id`) will be used instead as the `run_id` of the target run
- The `monitoring` namespace in the source run is copied to the `runtime` namespace in the target run
- All `sys` attributes except `state`, `description`, `name`, `custom_run_id`, `tags`, and `group_tags` are copied to the `legacy_sys` namespace
- `sys/running_time` and `sys/monitoring_time` are copied to `legacy_sys/running_time` and `legacy_sys/monitoring_time` respectively as floats in seconds
- `sys/size` is copied to `legacy_sys/size` as an integer in bytes

### Data that isn't copied
- Project description and members
- Project and model metadata
- Artifacts*
- FileSet (including source code)*
- All FileSeries objects*
  - Once FileSeries are supported in Neptune 3.x, the custom steps and descriptions of files in FileSeries still won't be copied
  - Run state: `sys/state`
  - Git info: `source_code/git`

\* Support for these will be added in upcoming releases.


---

## Post-migration checklist
- [ ] Review the logs for any errors. They also contain the URLs of both the source and target runs.  
- [ ] Add users to the new project and assign them relevant roles.
- [ ] Add project description.
- [ ] Recreate [saved views](https://docs.neptune.ai/runs_table/#custom-views), [dashboards](https://docs.neptune.ai/custom_dashboard/), and [reports](https://docs.neptune.ai/reports/) in the new project.
- [ ] Delete the `.tmp_*` folder in the working directory after verifying the migration.

---

## Troubleshooting

- **Project not found:** Ensure the `--legacy-project` is in the correct format and exists in your legacy workspace.
- **Authentication errors:** Double-check your API tokens and permissions in both the source and target projects.
- **Partial migration:** Check the log file for errors and rerun the script as needed.
- **Performance:** Increase `--max-workers` for faster migration, but be mindful of API rate limits.

---

## FAQ

**Q: Can I rerun the script if it fails?**  
A: Yes, runs already migrated will not be duplicated.

**Q: Are artifacts and files copied?**  
A: Not yet. See the [Notes and Limitations](#notes-and-limitations) section for details.

**Q: How do I filter which runs to migrate?**  
A: Use the `--query` argument with [NQL syntax](https://docs-legacy.neptune.ai/usage/nql/).

---

## Documentation

- [Neptune 2.x documentation](https://docs-legacy.neptune.ai/)
- [Neptune 3.x documentation](https://docs.neptune.ai/)
- [NQL query syntax](https://docs-legacy.neptune.ai/usage/nql/)

---

## Support and feedback
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
