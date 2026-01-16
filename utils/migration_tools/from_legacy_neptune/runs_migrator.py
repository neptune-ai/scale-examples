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
# https://github.com/neptune-ai/scale-examples/blob/main/utils/migration_tools/from_legacy_neptune/README.md

import argparse
import base64
import contextlib
import functools
import json
import logging
import os
import sys
import threading
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from typing import Optional

import neptune.metadata_containers
from neptune import management
from neptune.exceptions import MetadataInconsistency, MissingFieldException
from neptune_scale import Run
from neptune_scale.exceptions import (
    NeptuneAttributePathNonWritable,
    NeptuneSeriesStepNonIncreasing,
    NeptuneSeriesTimestampDecreasing,
)
from neptune_scale.projects import create_project
from neptune_scale.types import File
from tqdm.auto import tqdm

for logger in [
    "httpx",
    "urllib3",
    "requests",
    "neptune",
    "azure.core.pipeline.policies.http_logging_policy",
    "neptune_scale",
]:
    logging.getLogger(logger).setLevel(logging.ERROR)

READ_ONLY_NAMESPACES = {
    "sys/id",
    "sys/modification_time",
    "sys/monitoring_time",
    "sys/name",
    "sys/owner",
    "sys/ping_time",
    "sys/running_time",
    "sys/size",
    "sys/trashed",
}

UNCOPYABLE_NAMESPACES = {
    "sys/creation_time",
    "sys/state",
    "source_code/git",
}


def map_namespace(namespace: str) -> str:
    if namespace in READ_ONLY_NAMESPACES:
        return namespace.replace("sys", "legacy_sys")
    if namespace.startswith("monitoring/"):
        return "runtime/" + namespace[len("monitoring/") :]
    return namespace


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


def log_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            with contextlib.suppress(MissingFieldException):
                return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Failed to copy {args[-1]}/{args[1]} due to exception:\n{e}")

    return wrapper


def flatten_namespaces(
    dictionary: dict, prefix: Optional[list] = None, result: Optional[list] = None
) -> list:
    if prefix is None:
        prefix = []
    if result is None:
        result = []

    for k, v in dictionary.items():
        if isinstance(v, dict):
            flatten_namespaces(v, prefix + [k], result)
        elif prefix_str := "/".join(prefix):
            result.append(f"{prefix_str}/{k}")
        else:
            result.append(k)
    return result


def setup_logging(project: str) -> tuple[logging.Logger, str]:
    now = datetime.now()
    log_filename = now.strftime(f"{project.replace('/', '_')}_migration_%Y%m%d%H%M%S.log")

    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(funcName)20s:%(lineno)-4d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    logger = logging.getLogger(__name__)

    print(f"Logs available at {os.path.abspath(log_filename)}\n")

    return logger, log_filename


def exc_handler(logger: logging.Logger, exctype, value, tb):
    logger.exception("".join(traceback.format_exception(exctype, value, tb)))


def create_temporary_directory(log_filename: str) -> str:
    tmpdirname = os.path.abspath(
        os.path.join(os.getcwd(), f".tmp_{log_filename.replace('.log', '')}")
    )
    os.mkdir(tmpdirname)

    return tmpdirname


def validate_source_project(legacy_token: str, project: str):
    if project not in management.get_project_list(api_token=legacy_token):
        print(
            f"ERROR: Source project {project} does not exist. Please check project name",
            file=sys.stderr,
        )
        sys.exit(1)


def fetch_project_metadata(legacy_token: str, project: str) -> dict:
    with neptune.init_project(
        project=project,
        api_token=legacy_token,
        mode="read-only",
    ) as legacy_project:
        legacy_project_details = {
            "key": legacy_project["sys/id"].fetch(),
            "visibility": legacy_project["sys/visibility"].fetch(),
            "url": legacy_project.get_url(),
        }
        if legacy_project_details["visibility"] == "public":
            legacy_project_details["visibility"] = "pub"
        elif legacy_project_details["visibility"] == "private":
            legacy_project_details["visibility"] = "priv"

        return legacy_project_details


def fetch_source_runs(legacy_token: str, project: str, query: Optional[str] = None) -> list[str]:
    with neptune.init_project(
        project=project,
        api_token=legacy_token,
        mode="read-only",
    ) as source_project:
        return source_project.fetch_runs_table(columns=[], query=query).to_pandas()["sys/id"].values


@log_error
def copy_artifacts(source_run, namespace, run, id):
    # TODO: Not implemented in the new Neptune API
    return
    for artifact_location in [
        artifact.metadata["location"] for artifact in source_run[namespace].fetch_files_list()
    ]:
        run[map_namespace(namespace)].track_files(artifact_location)


@log_error
def copy_stringset(source_run, namespace, run, id):
    with contextlib.suppress(MissingFieldException, MetadataInconsistency):
        run.log_configs({map_namespace(namespace): source_run[namespace].fetch()})


@log_error
def copy_float_series(source_run, namespace, run, id):
    for row in source_run[namespace].fetch_values(progress_bar=False).itertuples():
        run.log_metrics(
            {map_namespace(namespace): row.value},
            step=row.step,
            timestamp=row.timestamp,
        )


@log_error
def copy_string_series(source_run, namespace, run, id):
    for row in source_run[namespace].fetch_values(progress_bar=False).itertuples():
        run.log_string_series(
            {map_namespace(namespace): row.value},
            step=row.step,
            timestamp=row.timestamp,
        )


@log_error
def copy_file(source_run, namespace, run, localpath, id):
    ext = source_run[namespace].fetch_extension()

    path = os.sep.join(namespace.split("/")[:-1])
    _download_path = os.path.join(localpath, path)
    os.makedirs(_download_path, exist_ok=True)
    source_run[namespace].download(_download_path, progress_bar=False)
    run.assign_files({map_namespace(namespace): f"{os.path.join(localpath, namespace)}.{ext}"})


@log_error
def copy_fileset(source_run, namespace, run, localpath, id):
    # TODO: Not implemented in the new Neptune API
    return
    _download_path = os.path.join(localpath, namespace)
    os.makedirs(_download_path, exist_ok=True)
    source_run[namespace].download(_download_path, progress_bar=False)

    _zip_path = os.path.join(_download_path, f"{namespace.split('/')[-1]}.zip")
    with zipfile.ZipFile(_zip_path) as zip_ref:
        zip_ref.extractall(_download_path)
    os.remove(_zip_path)

    with threadsafe_change_directory(_download_path):
        run[map_namespace(namespace)].upload_files(
            "*",
            wait=True,
        )


@log_error
def copy_fileseries(source_run, namespace, run, localpath, id):
    _download_path = os.path.join(localpath, namespace)
    source_run[namespace].download(_download_path, progress_bar=False)
    for step, file in enumerate(glob(f"{_download_path}{os.sep}*")):
        run.log_files(
            {map_namespace(namespace): File(file)},
            step=step,
        )


@log_error
def copy_atom(source_run, namespace, run, id):
    run.log_configs({map_namespace(namespace): source_run[namespace].fetch()})


def copy_metadata(
    source_run: neptune.Run,
    object_id: str,
    target_run: neptune.Run,
    tmpdirname: str,
) -> None:
    namespaces = flatten_namespaces(source_run.get_structure())

    _local_path = os.path.join(tmpdirname, object_id)

    for namespace in namespaces:
        if namespace in UNCOPYABLE_NAMESPACES:
            continue

        mapped_namespace = map_namespace(namespace)

        if namespace in READ_ONLY_NAMESPACES:
            # Create legacy_sys namespaces for read-only sys namespaces
            target_run.log_configs({mapped_namespace: source_run[namespace].fetch()})

        elif str(source_run[namespace]).startswith("<Artifact"):
            copy_artifacts(source_run, namespace, target_run, object_id)

        elif str(source_run[namespace]).startswith("<StringSet"):
            copy_stringset(source_run, namespace, target_run, object_id)

        elif str(source_run[namespace]).startswith("<FloatSeries"):
            copy_float_series(source_run, namespace, target_run, object_id)

        elif str(source_run[namespace]).startswith("<StringSeries"):
            copy_string_series(source_run, namespace, target_run, object_id)

        elif str(source_run[namespace]).startswith("<FileSet"):
            copy_fileset(source_run, namespace, target_run, _local_path, object_id)

        elif str(source_run[namespace]).startswith("<FileSeries"):
            copy_fileseries(source_run, namespace, target_run, _local_path, object_id)

        elif str(source_run[namespace]).startswith("<File"):
            copy_file(source_run, namespace, target_run, _local_path, object_id)

        else:
            copy_atom(source_run, namespace, target_run, object_id)

    target_run.close()


def init_target_run(
    source_run: neptune.Run, args: argparse.Namespace, logger: logging.Logger
) -> Run:
    def _error_callback(exc: BaseException, ts: Optional[float]) -> None:
        if isinstance(
            exc,
            (
                NeptuneSeriesStepNonIncreasing,
                NeptuneSeriesTimestampDecreasing,
                NeptuneAttributePathNonWritable,
            ),
        ):
            logger.warning(
                f"Encountered {exc.__class__.__name__} error while copying {source_run['sys/id'].fetch()}. Skipping affected attribute."
            )
        else:
            logger.exception(
                f"""Encountered error while copying {source_run["sys/id"].fetch()}:
{exc}
Aborting copy of the run.
                """
            )
            target_run.close()

    custom_run_id = source_run["sys/id"].fetch()
    with contextlib.suppress(MissingFieldException):
        custom_run_id = source_run["sys/custom_run_id"].fetch()

    target_run = Run(
        run_id=custom_run_id,
        project=(
            f"{args.new_workspace}/{args.new_project}" if args.new_project else args.legacy_project
        ),
        api_token=args.new_token,
        creation_time=source_run["sys/creation_time"].fetch(),
        enable_console_log_capture=False,
        on_error_callback=_error_callback,
    )
    return target_run


def copy_run(source_run_id: str, args: argparse.Namespace, logger: logging.Logger, tmpdirname: str):
    with neptune.init_run(
        project=args.legacy_project,
        api_token=args.legacy_token,
        with_id=source_run_id,
        mode="read-only",
    ) as source_run:
        with init_target_run(source_run, args, logger) as target_run:
            copy_metadata(source_run, source_run_id, target_run, tmpdirname)
            logger.info(f"Copied {source_run.get_url()} to {target_run.get_run_url()}")


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Neptune runs from a legacy workspace to a new workspace."
    )
    parser.add_argument(
        "--legacy-token", type=str, required=True, help="API token for the legacy workspace"
    )
    parser.add_argument(
        "--new-token", type=str, required=True, help="API token for the new workspace"
    )
    parser.add_argument(
        "--legacy-project",
        type=str,
        required=True,
        help="Legacy project name (in WORKSPACE_NAME/PROJECT_NAME format)",
    )
    parser.add_argument(
        "-w",
        "--new-workspace",
        type=str,
        required=True,
        help="New workspace name",
    )
    parser.add_argument(
        "--new-project",
        type=str,
        default=None,
        help="The project will be created if it doesn't already exist. If not provided, the project name will be the same as the legacy project name.",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Query filter for runs to be copied (Syntax: https://docs-legacy.neptune.ai/usage/nql/)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers to use (default: ThreadPoolExecutor's default)",
    )
    return parser.parse_args()


def main():
    args = fetch_args()

    validate_source_project(args.legacy_token, args.legacy_project)

    logger, log_filename = setup_logging(args.legacy_project)
    sys.excepthook = functools.partial(exc_handler, logger)

    legacy_project_details = fetch_project_metadata(args.legacy_token, args.legacy_project)
    create_project(
        api_token=args.new_token,
        workspace=args.new_workspace,
        name=args.new_project or args.legacy_project.split("/")[1],
        key=None if args.new_project else legacy_project_details["key"],
        visibility=legacy_project_details["visibility"],
    )
    if args.query:
        logger.info(f"Filter query: '{args.query}'")
    logger.info(f"Source project URL: {legacy_project_details['url']}runs")
    scale_instance_url = json.loads(base64.urlsafe_b64decode(args.new_token))["api_url"]
    logger.info(
        f"Target project URL: {scale_instance_url}/o/{args.new_workspace}/org/{args.new_project or args.legacy_project.split('/')[1]}/runs"
    )

    source_run_ids = fetch_source_runs(args.legacy_token, args.legacy_project, args.query)
    logger.info(f"Found {len(source_run_ids)} runs to copy")

    tmpdirname = create_temporary_directory(log_filename)
    logger.info(f"Temporary directory created at {tmpdirname}")

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_run = {
                executor.submit(copy_run, source_run_id, args, logger, tmpdirname): source_run_id
                for source_run_id in source_run_ids
            }

            for future in tqdm(as_completed(future_to_run), total=len(source_run_ids)):
                source_run_id = future_to_run[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Failed to copy {source_run_id} due to exception:\n{e}")

            logger.info("Export complete!")
            print("\nDone!")
    except Exception as e:
        logger.exception(f"Error during export: {e}")
        print("\nError!", file=sys.stderr)
        raise e

    finally:
        logging.shutdown()
        print(f"Check logs at {os.path.abspath(log_filename)}")


if __name__ == "__main__":
    main()
