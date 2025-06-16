# W&B to Neptune migration tool
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

__version__ = "0.2.0"

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from typing import Optional

import wandb
from neptune_scale import Run
from neptune_scale.exceptions import (
    NeptuneSeriesStepNonIncreasing,
    NeptuneSeriesTimestampDecreasing,
)
from neptune_scale.projects import create_project
from neptune_scale.types import File
from tqdm.auto import tqdm

SUPPORTED_DATATYPES = [int, float, str, datetime, bool, list, set]

EXCLUDED_PATHS = {"artifact/", "config.yaml", "media/", "wandb-"}


def exc_handler(exctype, value, tb):
    logger.exception("".join(traceback.format_exception(exctype, value, tb)))


sys.excepthook = exc_handler

# Configure logging to silence specific messages
logger = None
for logger_name in ["httpx", "urllib3", "wandb", "neptune_scale", "neptune"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def setup_logging(log_level: str) -> tuple[logging.Logger, str]:
    now = datetime.now()
    log_filename = now.strftime("wandb_to_neptune_%Y%m%d%H%M%S.log")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(funcName)20s:%(lineno)-4d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        force=True,
    )

    logger = logging.getLogger(__name__)

    print(f"Logs available at {log_filename}\n")

    return logger, log_filename


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


def copy_run(wandb_run, wandb_project_name: str) -> None:  # type: ignore
    def _error_callback(exc: BaseException, ts: Optional[float]) -> None:
        if isinstance(exc, (NeptuneSeriesStepNonIncreasing, NeptuneSeriesTimestampDecreasing)):
            logger.warning(f"Encountered {exc} error while copying {wandb_run.name}")
        else:
            neptune_run.close()

    logger.debug(f"Starting copy of run {wandb_run.project}/{wandb_run.name} (id: {wandb_run.id})")

    try:
        with Run(
            run_id=wandb_run.id,
            project=f"{neptune_workspace}/{wandb_project_name}",
            experiment_name=wandb_run.name,
            creation_time=datetime.fromisoformat(wandb_run.created_at.replace("Z", "+00:00")),
            on_error_callback=_error_callback,
            enable_console_log_capture=False,
        ) as neptune_run:
            # Add run description
            if wandb_run.notes:
                logger.debug(f"Logging run description for {wandb_run.name}")
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
                        logger.debug(f"Adding group tags for {wandb_run.name}")
                        neptune_run.add_tags([wandb_run.group], group_tags=True)
                    elif attr == "config":
                        logger.debug(f"Copying config for {wandb_run.name}")
                        copy_config(neptune_run, wandb_run)
                    elif attr == "tags":
                        logger.debug(f"Adding tags for {wandb_run.name}")
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

            try:
                copy_summary(neptune_run, wandb_run)
                copy_metrics(neptune_run, wandb_run)
                copy_system_metrics(neptune_run, wandb_run)
                download_folder = os.path.join(tmpdirname, wandb_run.project, wandb_run.id)
                copy_files(neptune_run, wandb_run, download_folder)
            except Exception as e:
                logger.error(f"Failed to copy {wandb_run.name} due to exception:\n{e}")
            else:
                logger.info(f"Copied {wandb_run.url} to {neptune_run.get_run_url()}")

    finally:
        logger.debug(
            f"Finished copy of run {wandb_run.project}/{wandb_run.name} (id: {wandb_run.id})"
        )


def copy_config(neptune_run: Run, wandb_run) -> None:  # type: ignore
    flat_config = stringify_unsupported(wandb_run.config)

    try:
        for key, value in flat_config.items():
            neptune_run.log_configs({f"config/{key}": value})
    except Exception as e:
        logger.error(
            f"Failed to copy config {key} from W&B run {wandb_run.name} due to exception:\n{e}"
        )


def copy_summary(neptune_run: Run, wandb_run) -> None:  # type: ignore
    logger.debug(f"Copying summary for {wandb_run.name}")
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
            continue
        except Exception as e:
            logger.error(
                f"Failed to copy summary {key} from W&B run {wandb_run.name} due to exception:\n{e}"
            )


def copy_metrics(neptune_run: Run, wandb_run) -> None:  # type: ignore
    logger.debug(f"Copying metrics for {wandb_run.name}")
    for record in wandb_run.scan_history():
        for key, value in record.items():
            if (
                value is None
                or key.startswith("_")
                or (isinstance(value, dict) and value["_type"] == "table-file")
            ):
                continue
            try:
                neptune_run.log_metrics(
                    {key: value},
                    step=record.get("epoch") or record.get("_step"),
                    timestamp=(
                        datetime.fromtimestamp(record.get("_timestamp"))
                        if record.get("_timestamp")
                        else None
                    ),
                )
            except Exception as e:
                logger.error(
                    f"Failed to copy metric {key} from W&B run {wandb_run.name} due to exception:\n{e}"
                )


def copy_system_metrics(neptune_run: Run, wandb_run) -> None:  # type: ignore
    logger.debug(f"Copying system metrics for {wandb_run.name}")
    for step, record in enumerate(wandb_run.history(stream="system", pandas=False)):
        for key in record:
            if key.startswith("_"):  # Excluding '_runtime', '_timestamp', '_wandb'
                continue

            value = record.get(key)
            if value is None:
                continue
            try:
                neptune_run.log_metrics(
                    {f"runtime/{key.replace('system.', '').replace('.', '/')}": value},
                    step=step,
                    timestamp=datetime.fromtimestamp(record.get("_timestamp")),
                )
            except Exception as e:
                logger.error(
                    f"Failed to copy system metric {key} from W&B run {wandb_run.name} due to exception:\n{e}"
                )


def copy_console_output(neptune_run: Run, download_path: str) -> None:  # type: ignore
    with open(download_path, encoding="utf-8", errors="ignore") as f:
        for step, line in enumerate(f):
            neptune_run.log_string_series({"runtime/stdout": line}, step=step)


def copy_source_code(
    neptune_run: Run,
    download_path: str,
    filename: str,
) -> None:
    abs_path = os.path.abspath(download_path)
    rel_filename = filename.replace("code/", "")
    neptune_run.assign_files(
        {f"source_code/files/{rel_filename}": File(source=abs_path, mime_type="text/plain")}
    )


def copy_requirements(neptune_run: Run, download_path: str) -> None:
    neptune_run.assign_files(
        {"source_code/requirements.txt": File(source=download_path, mime_type="text/plain")}
    )


def copy_other_files(neptune_run: Run, download_path: str, filename: str, namespace: str) -> None:
    abs_path = os.path.abspath(download_path)
    neptune_run.assign_files({f"{namespace}/{filename}": File(source=abs_path)})


def copy_files(neptune_run: Run, wandb_run, download_folder: str) -> None:  # type: ignore
    for file in wandb_run.files():
        if file.size and not any(file.name.startswith(path) for path in EXCLUDED_PATHS):
            download_path = os.path.join(download_folder, file.name)
            try:
                logger.debug(
                    f"Downloading file {file.name} for run {wandb_run.name} to {download_path}"
                )
                file.download(root=download_folder, replace=True, exist_ok=True)
                if file.name == "output.log":
                    logger.debug(f"Copying console output for {wandb_run.name}")
                    copy_console_output(neptune_run, download_path)

                elif file.name.startswith("code/"):
                    logger.debug(f"Copying source code file {file.name} for {wandb_run.name}")
                    copy_source_code(neptune_run, download_path, file.name)

                elif file.name == "requirements.txt":
                    logger.debug(f"Copying requirements.txt for {wandb_run.name}")
                    copy_requirements(neptune_run, download_path)

                elif "ckpt" in file.name or "checkpoint" in file.name:
                    logger.debug(f"Copying checkpoint file {file.name} for {wandb_run.name}")
                    copy_other_files(neptune_run, download_path, file.name, namespace="checkpoints")

                else:
                    logger.debug(f"Copying other file {file.name} for {wandb_run.name}")
                    copy_other_files(neptune_run, download_path, file.name, namespace="files")
            except Exception as e:
                logger.error(
                    f"Failed to copy {download_path} for {wandb_run.name} due to exception:\n{e}"
                )


def copy_project(wandb_project) -> None:  # type: ignore
    wandb_project_name = wandb_project.name.replace("_", "-")

    logger.debug(f"Starting copy of project {wandb_project.name}")

    # Create a new Neptune project for each W&B project
    create_project(
        name=f"{neptune_workspace}/{wandb_project_name}",
        description=f"Exported from {wandb_project.url}",
    )

    wandb_runs = [
        run for run in client.runs(f"{wandb_entity}/{wandb_project.name}")
    ]  # sourcery skip: identity-comprehension

    FAILED_RUNS = []

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
            wandb_run = future_to_run[future]
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                logger.error(f"Timeout copying run {wandb_run.project}/{wandb_run.name}")
                FAILED_RUNS.append((wandb_run.project, wandb_run.name, "Timeout"))
            except Exception as e:
                logger.exception(
                    f"Failed to copy {wandb_run.project}/{wandb_run.name} due to exception:\n{e}"
                )
                FAILED_RUNS.append((wandb_run.project, wandb_run.name, str(e)))

    if FAILED_RUNS:
        logger.error(f"Some runs failed to copy in project {wandb_project.name}: {FAILED_RUNS}")
        print(f"Some runs failed to copy in project {wandb_project.name}: {FAILED_RUNS}")

    logger.debug(f"Finished copy of project {wandb_project.name}")


if __name__ == "__main__":
    client = wandb.Api(timeout=120)
    parser = argparse.ArgumentParser(description="Migrate W&B runs to Neptune.")
    parser.add_argument(
        "--wandb-entity", type=str, default=client.default_entity, help="W&B entity name"
    )
    parser.add_argument(
        "--neptune-workspace",
        type=str,
        default=(
            os.getenv("NEPTUNE_PROJECT").split("/")[0] if os.getenv("NEPTUNE_PROJECT") else None
        ),
        help="Neptune workspace name",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument(
        "--projects",
        type=str,
        default=None,
        help="Comma-separated list of projects to copy. Leave blank to copy all projects in the entity.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each run/project (default: 600)",
    )
    args = parser.parse_args()

    wandb_entity = args.wandb_entity
    neptune_workspace = args.neptune_workspace
    num_workers = args.num_workers
    timeout = args.timeout

    if not neptune_workspace:
        raise ValueError(
            "Neptune workspace not found. Either set the `NEPTUNE_PROJECT` environment variable or provide it via the `--neptune-workspace` CLI argument."
        )

    logger, log_filename = setup_logging(args.log_level)
    logger.info(f"Running version {__version__}")
    logger.info(f"Copying from W&B entity {wandb_entity} to Neptune workspace {neptune_workspace}")

    # Create temporary directory to store local metadata
    tmpdirname = os.path.abspath(
        os.path.join(os.getcwd(), "tmp_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    )
    os.makedirs(tmpdirname, exist_ok=True)
    logger.info(f"Temporary directory created at {tmpdirname}")

    wandb_projects = [
        project for project in client.projects()
    ]  # sourcery skip: identity-comprehension
    wandb_project_names = [project.name for project in wandb_projects]

    if args.projects:
        selected_projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    else:
        selected_projects = wandb_project_names

    if not_found := set(selected_projects) - set(wandb_project_names):
        print(f"Projects not found: {not_found}")
        logger.warning(f"Projects not found: {not_found}")
        selected_projects = set(selected_projects) - not_found

    print(f"Copying {len(selected_projects)} projects: {selected_projects}\n")
    logger.info(f"Copying {len(selected_projects)} projects: {selected_projects}")

    FAILED_PROJECTS = []
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
                wandb_project = future_to_project[future]
                try:
                    future.result(timeout=timeout)
                except TimeoutError:
                    logger.error(f"Timeout copying project {wandb_project.name}")
                    FAILED_PROJECTS.append((wandb_project.name, "Timeout"))
                except Exception as e:
                    logger.exception(f"Failed to copy {wandb_project.name} due to exception:\n{e}")
                    FAILED_PROJECTS.append((wandb_project.name, str(e)))

        if FAILED_PROJECTS:
            logger.error(f"Some projects failed to copy: {FAILED_PROJECTS}")
            print(f"Some projects failed to copy: {FAILED_PROJECTS}")
            sys.exit(1)

        logger.info("Copy complete!")
        print("\nDone!")

    except Exception as e:
        logger.exception(f"Copy failed due to exception:\n{e}")
        print("\nError!")
        raise e

    finally:
        logging.shutdown()
        print(f"Check logs at {log_filename}\n")
