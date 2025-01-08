#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Instructions on how to use this script can be found at
# https://github.com/neptune-ai/scale-examples/blob/main/utils/migration_tools/from_wandb/README.md

# %%
import functools
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime

import wandb
from neptune_scale import Run
from neptune_scale.projects import create_project
from tqdm.auto import tqdm

SUPPORTED_DATATYPES = [int, float, str, datetime, bool, list, set]

if __name__ == "__main__":
    # %%
    client = wandb.Api(timeout=120)

    # %% Input prompts

    if client.default_entity:
        wandb_entity = (
            input(
                f"Enter W&B entity name. Leave blank to use the default entity ({client.default_entity}): "
            ).strip()
            or client.default_entity
        )
    else:
        wandb_entity = input("Enter W&B entity name: ").strip()

    if default_neptune_workspace := os.getenv("NEPTUNE_PROJECT"):
        default_neptune_workspace = default_neptune_workspace.split("/")[0]
        neptune_workspace = (
            input(
                f"Enter Neptune workspace name. Leave blank to use the default workspace ({default_neptune_workspace}): "
            ).strip()
            or default_neptune_workspace
        )
    else:
        neptune_workspace = input("Enter Neptune workspace name: ").strip()

    num_workers = input(
        "Enter the number of workers to use (int). Leave empty to determine automatically: "
    ).strip()

    num_workers = None if num_workers == "" else int(num_workers)

    # %% Setup logging
    now = datetime.now()
    log_filename = now.strftime("wandb_to_neptune_%Y%m%d%H%M%S.log")

    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    logger = logging.getLogger(__name__)

    print(f"Logs available at {log_filename}\n")

    # Silencing Neptune and httpx messages, W&B errors, urllib connection pool warnings
    logging.getLogger("neptune").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("wandb").setLevel(logging.ERROR)

    def exc_handler(exctype, value, tb):
        logger.exception("".join(traceback.format_exception(exctype, value, tb)))

    sys.excepthook = exc_handler

    logger.info(f"Copying from W&B entity {wandb_entity} to Neptune workspace {neptune_workspace}")

    # %% Create temporary directory to store local metadata
    # Not needed for Neptune Scale as file upload is not supported
    # tmpdirname = os.path.abspath(
    #     os.path.join(os.getcwd(), "tmp_" + now.strftime("%Y%m%d%H%M%S"))
    # )
    # os.makedirs(tmpdirname, exist_ok=True)
    # logger.info(f"Temporary directory created at {tmpdirname}")

    # %%
    wandb_projects = [
        project for project in client.projects()
    ]  # sourcery skip: identity-comprehension
    wandb_project_names = [project.name for project in wandb_projects]

    print(f"W&B projects found ({len(wandb_project_names)}): {wandb_project_names}")

    selected_projects = input(
        "Enter projects you want to copy (comma-separated). Leave blank to copy all projects:  "
    ).strip()

    # %%
    if selected_projects == "":
        selected_projects = wandb_project_names
    else:
        selected_projects = [project.strip() for project in selected_projects.split(",")]

    if not_found := set(selected_projects) - set(wandb_project_names):
        print(f"Projects not found: {not_found}")
        logger.warning(f"Projects not found: {not_found}")
        selected_projects = set(selected_projects) - not_found

    print(f"Copying {len(selected_projects)} projects: {selected_projects}\n")
    logger.info(f"Copying {len(selected_projects)} projects: {selected_projects}")

    # %% UDFs
    def stringify_unsupported(d, parent_key="", sep="/"):
        items = {}
        if not isinstance(d, (dict, list, tuple, set)):
            return d if type(d) in SUPPORTED_DATATYPES else str(d)
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list, tuple, set)):
                    items |= stringify_unsupported(v, new_key, sep=sep)
                else:
                    items[new_key] = v if type(v) in SUPPORTED_DATATYPES else str(v)
        elif isinstance(d, (list, tuple, set)):
            for i, v in enumerate(d):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list, tuple, set)):
                    items.update(stringify_unsupported(v, new_key, sep=sep))
                else:
                    items[new_key] = v if type(v) in SUPPORTED_DATATYPES else str(v)
        return items

    def log_error(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"[{func.__name__}]\tFailed to copy '{args[1]}' from W&B run '{args[0].name}' due to exception:\n{e}"
                )

        return wrapper

    @contextmanager
    def threadsafe_change_directory(new_dir):
        lock = threading.Lock()
        old_dir = os.getcwd()
        try:
            with lock:
                os.chdir(new_dir)
            yield
        finally:
            with lock:
                os.chdir(old_dir)

    def copy_run(wandb_run: client.run, wandb_project_name: str) -> None:  # type: ignore
        with Run(
            run_id=wandb_run.id,
            project=f"{neptune_workspace}/{wandb_project_name}",
            experiment_name=wandb_run.name,
            creation_time=datetime.fromisoformat(wandb_run.created_at.replace("Z", "+00:00")),
        ) as neptune_run:
            # Add run description
            if wandb_run.notes:
                neptune_run.log_configs({"sys/description": wandb_run.notes})

            # Add W&B run attributes
            for attr in wandb_run._attrs:
                try:
                    if (
                        attr.startswith("user")
                        or attr.startswith("_")
                        or callable(getattr(wandb_run, attr))
                    ):
                        # Skip unsupported attributes
                        continue
                    if attr == "group":
                        # Add group tags
                        neptune_run.add_tags([wandb_run.group], group_tags=True)
                    elif attr == "config":
                        copy_config(neptune_run, wandb_run)
                    elif attr == "tags":
                        neptune_run.add_tags(wandb_run.tags)
                    elif isinstance(getattr(wandb_run, attr), dict):
                        for k, v in stringify_unsupported(getattr(wandb_run, attr)).items():
                            neptune_run.log_configs({f"wandb/{attr}/{k}": v})
                    else:
                        neptune_run.log_configs({f"wandb/{attr}": getattr(wandb_run, attr)})
                except TypeError:
                    pass
                except Exception as e:
                    logger.error(
                        f"[{copy_run.__name__}]\tFailed to copy '{attr}' from W&B run '{wandb_run.name}' due to exception:\n{e}"
                    )

            copy_summary(neptune_run, wandb_run)
            copy_metrics(neptune_run, wandb_run)
            copy_system_metrics(neptune_run, wandb_run)
            # copy_files(neptune_run, wandb_run)
            neptune_run.wait_for_processing()
            logger.info(
                f"Copied {wandb_run.url} to https://scale.neptune.ai/{neptune_run._project}/runs/table?viewId=standard-view&query=(%60sys%2Fname%60%3Astring%20MATCHES%20%22{wandb_run.name}%22)&experimentsOnly=false&runsLineage=FULL&lbViewUnpacked=true"
            )

    def copy_config(neptune_run: Run, wandb_run: client.run) -> None:  # type: ignore
        flat_config = stringify_unsupported(wandb_run.config)

        for key, value in flat_config.items():
            neptune_run.log_configs({f"config/{key}": value})

    def copy_summary(neptune_run: Run, wandb_run: client.run) -> None:  # type: ignore
        summary = wandb_run.summary
        for key, value in summary.items():
            if key.startswith("_") or (isinstance(value, wandb.old.summary.SummarySubDict)):
                continue
            try:
                stringified_summary = stringify_unsupported(value)
                if isinstance(stringified_summary, dict):
                    for k, v in stringified_summary.items():
                        neptune_run.log_configs({f"summary/{key}/{k}": v})
                else:
                    neptune_run.log_configs({f"summary/{key}": stringified_summary})
            except KeyError:
                print(f"KeyError: {key}")
                continue

    def copy_metrics(neptune_run: Run, wandb_run: client.run) -> None:  # type: ignore
        for record in wandb_run.scan_history():
            for key, value in record.items():
                if (
                    value is None
                    or key.startswith("_")
                    or (isinstance(value, dict) and value["_type"] == "table-file")
                ):
                    continue

                neptune_run.log_metrics(
                    {key: value},
                    step=record.get("epoch") or record.get("_step"),
                    timestamp=(
                        datetime.fromtimestamp(record.get("_timestamp"))
                        if record.get("_timestamp")
                        else None
                    ),
                )

    def copy_system_metrics(neptune_run: Run, wandb_run: client.run) -> None:  # type: ignore
        for step, record in enumerate(wandb_run.history(stream="system", pandas=False)):
            for key in record:
                if key.startswith("_"):  # Excluding '_runtime', '_timestamp', '_wandb'
                    continue

                value = record.get(key)
                if value is None:
                    continue

                neptune_run.log_metrics(
                    {f"system/{key.replace('system.', '')}": value},
                    step=step,
                    timestamp=datetime.fromtimestamp(record.get("_timestamp")),
                )

    @log_error
    def copy_console_output(neptune_run: Run, download_path: str) -> None:  # type: ignore
        # with open(download_path) as f:
        #     for line in f:
        #         neptune_run["monitoring/stdout"].append(line)
        raise NotImplementedError("Console logging is not implemented in Neptune Scale yet")

    @log_error
    def copy_source_code(
        neptune_run: Run,
        download_path: str,
        filename: str,
    ) -> None:
        # with threadsafe_change_directory(
        #     os.path.join(download_path.replace(filename, ""), "code")
        # ):
        #     neptune_run["source_code/files"].upload_files(
        #         filename.replace("code/", ""), wait=True
        #     )
        raise NotImplementedError("Source code logging is not implemented in Neptune Scale yet")

    @log_error
    def copy_requirements(neptune_run: Run, download_path: str) -> None:
        # neptune_run["source_code/requirements"].upload(download_path)
        raise NotImplementedError("Requirements logging is not implemented in Neptune Scale yet")

    @log_error
    def copy_other_files(
        neptune_run: Run, download_path: str, filename: str, namespace: str
    ) -> None:
        # with threadsafe_change_directory(download_path.replace(filename, "")):
        #     neptune_run[namespace].upload_files(filename, wait=True)
        raise NotImplementedError("File logging is not implemented in Neptune Scale yet")

    def copy_files(neptune_run: Run, wandb_run: client.run) -> None:  # type: ignore
        # EXCLUDED_PATHS = {"artifact/", "config.yaml", "media/", "wandb-"}
        # download_folder = os.path.join(tmpdirname, wandb_run.project, wandb_run.id)
        # for file in wandb_run.files():
        #     if (
        #         file.size and not any(file.name.startswith(path) for path in EXCLUDED_PATHS)
        #     ):  # A zero-byte file will be returned even when the `output.log` file does not exist
        #         download_path = os.path.join(download_folder, file.name)
        #         try:
        #             file.download(root=download_folder, replace=True, exist_ok=True)
        #             if file.name == "output.log":
        #                 copy_console_output(neptune_run, download_path)

        #             elif file.name.startswith("code/"):
        #                 copy_source_code(neptune_run, download_path, file.name)

        #             elif file.name == "requirements.txt":
        #                 copy_requirements(neptune_run, download_path)

        #             elif "ckpt" in file.name or "checkpoint" in file.name:
        #                 copy_other_files(
        #                     neptune_run, download_path, file.name, namespace="checkpoints"
        #                 )

        #             else:
        #                 copy_other_files(
        #                     neptune_run, download_path, file.name, namespace="files"
        #                 )
        #         except Exception as e:
        #             logger.error(f"Failed to copy {download_path} due to exception:\n{e}")
        raise NotImplementedError("File logging is not implemented in Neptune Scale yet")

    def copy_project(wandb_project: client.project) -> None:  # type: ignore
        # sourcery skip: identity-comprehension
        wandb_project_name = wandb_project.name.replace("_", "-")

        # Create a new Neptune project for each W&B project
        create_project(
            name=f"{neptune_workspace}/{wandb_project_name}",
            description=f"Exported from {wandb_project.url}",
        )

        wandb_runs = [run for run in client.runs(f"{wandb_entity}/{wandb_project.name}")]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_run = {
                executor.submit(copy_run, wandb_run, wandb_project_name): wandb_run
                for wandb_run in wandb_runs
            }

            for future in tqdm(
                as_completed(future_to_run),
                total=len(future_to_run),
                desc=f"Copying {wandb_project_name} runs",
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(
                        f"Failed to copy {future_to_run[future].project}/{future_to_run[future].name} due to exception:\n{e}"
                    )

    # %%
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_project = {
                executor.submit(copy_project, wandb_project): wandb_project
                for wandb_project in wandb_projects
                if wandb_project.name in selected_projects
            }

            for future in tqdm(
                as_completed(future_to_project),
                total=len(future_to_project),
                desc="Copying projects",
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(
                        f"Failed to copy {future_to_project[future]} due to exception:\n{e}"
                    )

        logger.info("Copy complete!")
        print("\nDone!")

    except Exception as e:
        logger.exception(f"Copy failed due to exception:\n{e}")
        print("\nError!")
        raise e

    finally:
        logging.shutdown()
        print(f"Check logs at {log_filename}\n")
