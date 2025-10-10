import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

__version__ = "0.1.1"

# TODO: Support runs mode
# TODO: Add support for audio and html
# TODO: Clean download directory

# Neptune integration
try:
    import neptune_query as nq
    from neptune_query.filters import Filter

    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

_LOGO_PATH = "utils/visualization_tools/file_comparison_app/assets/neptune_ai_signet_color.png"

# Configure page
st.set_page_config(
    page_title="File Comparison App",
    page_icon=_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://support.neptune.ai/",
        "Report a bug": "https://github.com/neptune-ai/scale-examples/issues",
        "About": f"**File Comparison App** v{__version__}\n\nVisualize and compare media file-series across different Neptune experiments.",
    },
)
st.logo(_LOGO_PATH)

_SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
}
_SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}
_SUPPORTED_EXTENSIONS = _SUPPORTED_IMAGE_EXTENSIONS | _SUPPORTED_VIDEO_EXTENSIONS


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0


def is_media_file(file_path: str) -> bool:
    """Check if file is a static image, gif, or video"""
    return Path(file_path).suffix.lower() in _SUPPORTED_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """Check if file is a video"""
    return Path(file_path).suffix.lower() in _SUPPORTED_VIDEO_EXTENSIONS


def create_file_statistics(files: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create file statistics DataFrame"""
    if not files:
        return pd.DataFrame()

    df = pd.DataFrame(files)
    df["modified_date"] = pd.to_datetime(df["modified"], unit="s")

    return df


@st.cache_data()
def download_neptune_files(
    project_name: str,
    experiment_regex: str,
    attribute_regex: str,
    download_dir: str,
    include_archived: bool,
) -> List[Dict[str, Any]]:
    """Download files from Neptune and return file information"""
    if not NEPTUNE_AVAILABLE:
        st.error("Neptune Query is not available. Please install neptune-query package.")
        return []

    try:
        # List experiments
        _filter = Filter.name(experiment_regex)
        if not include_archived:
            _filter = _filter & Filter.eq("sys/archived", False)

        exps = nq.list_experiments(project=project_name, experiments=_filter)

        if not exps:
            st.warning(f"No experiments found matching pattern: {experiment_regex}")
            return [], {}

        # Fetch files from experiments using the attribute regex
        files = nq.fetch_series(project=project_name, experiments=exps, attributes=attribute_regex)

        # Create project-specific download directory
        # Use project name as top-level folder to prevent mixing experiments from different projects
        project_download_dir = Path(download_dir) / project_name.replace("/", "_")
        project_download_dir.mkdir(parents=True, exist_ok=True)

        # Download files to project-specific directory
        # TODO: Download only supported file types
        nq.download_files(files=files, destination=str(project_download_dir))

        # Convert to our file format
        downloaded_files = []

        # Scan the project-specific download directory for files
        # Only include files from folders that match the experiment regex
        all_files = []
        for item in project_download_dir.iterdir():
            if item.is_dir() and re.search(experiment_regex, item.name):
                # Add all files from this matching folder
                all_files.extend(item.rglob("*.*"))

        media_count = 0
        for file_path in all_files:
            try:
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "relative_path": str(file_path.relative_to(project_download_dir)),
                    "size_mb": get_file_size_mb(str(file_path)),
                    "extension": file_path.suffix.lower(),
                    "is_media": is_media_file(str(file_path)),
                    "is_video": is_video_file(str(file_path)),
                    "modified": file_path.stat().st_mtime,
                }
                downloaded_files.append(file_info)

                if file_info["is_media"]:
                    media_count += 1

            except Exception as e:
                st.warning(f"Error processing file {file_path}: {e}")

        # Store download info for display in expander
        download_info = {
            "project_name": project_name,
            "experiments": exps,
            "attribute_regex": attribute_regex or ".*",
            "files_fetched": len(files),
            "download_dir": str(project_download_dir),
            "total_files": len(all_files),
            "media_files": media_count,
            "total_processed": len(downloaded_files),
        }

        return downloaded_files, download_info

    except Exception as e:
        st.error(f"Error downloading from Neptune: {e}")
        return [], {}


def main():
    st.title("Neptune File Comparison App")
    st.text("Visualize and compare media file series across different Neptune experiments")

    # Project configuration in expandable container
    st.sidebar.markdown(f"**Version:** {__version__}")

    if not NEPTUNE_AVAILABLE:
        st.error("Neptune Query not available. Install using `pip install -U neptune-query`")
        st.stop()

    with st.sidebar.expander("Neptune Configuration", icon=":material/settings:", expanded=True):
        # Neptune API token
        _neptune_api_token = st.session_state.get("neptune_api_token") or os.getenv(
            "NEPTUNE_API_TOKEN"
        )
        st.session_state.neptune_api_token = st.text_input(
            "Neptune API Token",
            value=_neptune_api_token,
            placeholder="your_api_token",
            type="password",
            help="Defaults to `NEPTUNE_API_TOKEN` environment variable",
            icon=":material/password:",
        )
        if st.session_state.neptune_api_token or _neptune_api_token:
            nq.set_api_token(st.session_state.neptune_api_token or _neptune_api_token)

        # Neptune project
        _neptune_project = st.session_state.get("neptune_project") or os.getenv("NEPTUNE_PROJECT")
        neptune_project = st.text_input(
            "Neptune Project",
            value=_neptune_project,
            placeholder="workspace_name/project_name",
            help="In the format `workspace_name/project_name`. Defaults to `NEPTUNE_PROJECT` environment variable.",
            icon=":material/folder:",
        )
        st.session_state.neptune_project = neptune_project

    with st.sidebar.expander("Download Configuration", icon=":material/tune:", expanded=True):
        # Download directory
        download_directory = st.text_input(
            "Download Directory",
            value=st.session_state.get("download_directory", "neptune_downloads"),
            help="Directory to download Neptune files to. Defaults to `neptune_downloads` in the current working directory.",
            icon=":material/folder:",
        )
        st.session_state.download_directory = download_directory

        # Experiment regex (required field)
        # TODO: Support passing a list of experiment names
        experiment_regex = st.text_input(
            "Experiments Regex",
            value=st.session_state.get("experiment_regex", ""),
            help="Regex specifying the experiments names to download from",
            placeholder="exp_.*",
            icon=":material/search:",
        )
        st.session_state.experiment_regex = experiment_regex

        include_archived = st.toggle("Include archived experiments", value=False)

        # Validate experiment regex is valid
        if not experiment_regex or not experiment_regex.strip():
            st.error(
                "Experiment regex is required. Please enter a pattern to match experiment names.",
                icon=":material/warning:",
            )
            experiment_regex_valid = False
        elif experiment_regex.strip() == ".*":
            st.warning(
                "Experiment regex is set to `.*`. This will download all experiments from the project.",
                icon=":material/warning:",
            )
            experiment_regex_valid = True
        else:
            experiment_regex_valid = True

        # Attribute regex
        attribute_regex = st.text_input(
            "Attribute Regex",
            value=st.session_state.get("attribute_regex"),
            help="Regex pattern to match file attribute names. Defaults to `None` (all attributes)",
            placeholder="image_.*",
            icon=":material/search:",
        )
        st.session_state.attribute_regex = attribute_regex

        if st.button(
            "Clear cache",
            icon=":material/delete:",
            width="stretch",
            help="Clear the cache to fetch latest files",
        ):
            st.cache_data.clear()
            st.rerun()

        if st.button(
            "Download and Visualize", icon=":material/download:", width="stretch", type="primary"
        ):
            # Check if experiment regex is valid before proceeding
            if not experiment_regex_valid:
                st.error("Cannot proceed: Experiment regex is required!", icon=":material/error:")
                st.stop()

            with st.spinner("Downloading files from Neptune...", show_time=True):
                files, download_info = download_neptune_files(
                    neptune_project,
                    experiment_regex,
                    attribute_regex,
                    download_directory,
                    include_archived,
                )
                st.session_state.files = files
                st.session_state.download_info = download_info
                st.session_state.directory_scanned = True

            # Show success/warning message
            if files:
                st.success(f"Successfully downloaded {len(files)} files", icon=":material/check:")
            else:
                st.warning("No files were downloaded. Check your configuration.")

        # Show download details in expander if available
        if "download_info" in st.session_state and st.session_state.download_info:
            with st.sidebar.expander(
                "Download Details", icon=":material/download:", expanded=False
            ):
                info = st.session_state.download_info
                st.write(f"**Project:** {info.get('project_name', 'N/A')}")
                with st.expander(
                    f"Experiments Found: **{len(info['experiments'])}**", icon=":material/science:"
                ):
                    for experiment in info["experiments"]:
                        st.write(experiment)
                st.write(f"**Attribute Regex:** `{info['attribute_regex']}`")
                st.write(f"**Download Directory:** {info['download_dir']}")
                st.write(f"**Total Files:** {info['total_files']}")
                st.write(f"**Files Processed:** {info['total_processed']}")
                st.write(f"**Media Files:** {info['media_files']}")

    # Gallery view options
    if "files" in st.session_state and st.session_state.files:
        # Apply regex filters to get experiments and media files

        experiment_pattern = st.session_state.get("experiment_regex", ".*")
        attribute_pattern = st.session_state.get("attribute_regex", ".*") or ".*"

        # Filter files by media type and regex patterns
        filtered_files = []
        for file_info in st.session_state.files:
            # Check if it's a media file
            if file_info.get("is_media", False):
                # Check experiment regex (folder name)
                path_parts = Path(file_info["relative_path"]).parts
                if len(path_parts) > 1:
                    experiment_name = path_parts[0]
                    if re.search(experiment_pattern, experiment_name) and re.search(
                        attribute_pattern, file_info["name"]
                    ):
                        filtered_files.append(file_info)

        # Get unique experiments from filtered files
        top_level_folders = set()
        for file_info in filtered_files:
            path_parts = Path(file_info["relative_path"]).parts
            if len(path_parts) > 1:
                top_level_folders.add(path_parts[0])

        # Create individual toggles for each experiment
        folder_toggles = {}
        if top_level_folders:
            with st.sidebar.expander(
                "Select experiments to view",
                icon=":material/visibility:",
                expanded=True,
            ):
                # Add select all / deselect all buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All", icon=":material/check_box:", width="stretch"):
                        for folder in top_level_folders:
                            st.session_state[f"folder_toggle_{folder}"] = True
                        st.rerun()

                with col2:
                    if st.button(
                        "Deselect All", icon=":material/check_box_outline_blank:", width="stretch"
                    ):
                        for folder in top_level_folders:
                            st.session_state[f"folder_toggle_{folder}"] = False
                        st.rerun()

                for folder in sorted(top_level_folders):
                    folder_toggles[folder] = st.checkbox(
                        folder,
                        value=True,  # Default to showing all folders
                        key=f"folder_toggle_{folder}",
                    )

        # Gallery layout controls
        st.sidebar.subheader("📄 Gallery Layout")

        # Layout orientation
        layout_orientation = st.sidebar.segmented_control(
            "Column headers", options=["Steps", "Experiments"], default="Steps", width="stretch"
        )

        # Pagination controls
        columns_per_page = st.sidebar.slider(
            "Columns per page",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of columns to show at once",
        )

        # Image sizing controls
        # st.sidebar.subheader("🖼️ Media Display")
        # consistent_sizing = st.sidebar.checkbox(
        #     "Consistent media size",
        #     value=False,
        #     help="Resize all images to the same dimensions for easier comparison",
        # )

        # if consistent_sizing:
        #     image_width = st.sidebar.slider(
        #         "Media width (pixels)", min_value=100, max_value=500, value=200, step=10
        #     )
        #     image_height = st.sidebar.slider(
        #         "Media height (pixels)", min_value=100, max_value=500, value=200, step=10
        #     )
        #     consistent_size = (image_width, image_height)
        # else:
        #     consistent_size = None

        st.session_state.filtered_files = filtered_files
        st.session_state.folder_toggles = folder_toggles
        st.session_state.images_per_page = columns_per_page
        # st.session_state.consistent_size = consistent_size
        st.session_state.layout_orientation = layout_orientation

    # Main content area
    if "files" not in st.session_state or not st.session_state.files:
        st.info(
            "Configure the download and click 'Download and Visualize' to get started",
            icon=":material/arrow_circle_left:",
        )
        return

    # Image comparison gallery

    filtered_df = create_file_statistics(
        st.session_state.get("filtered_files", st.session_state.files)
    )

    if not filtered_df.empty:
        # Grid gallery: rows = experiments, columns = steps
        media_files = [f for f in filtered_df.itertuples() if f.is_media]

        if media_files:
            st.subheader("Comparison Grid")

            # Get folder toggles, pagination settings, and image sizing
            folder_toggles = st.session_state.get("folder_toggles", {})
            columns_per_page = st.session_state.get("images_per_page", 3)
            consistent_size = st.session_state.get("consistent_size", None)
            layout_orientation = st.session_state.get("layout_orientation", "Steps")

            # Extract step number from filename or path
            def extract_step_number(file_info):
                try:
                    # Try to extract step from filename first
                    match = re.search(r"step_(\d+)", file_info.name)
                    if match:
                        return int(match.group(1))

                    # Try to extract step from path (for Neptune downloads)
                    match = re.search(r"step[_-]?(\d+)", file_info.path)
                    if match:
                        return int(match.group(1))

                    # Try to extract step from relative path
                    match = re.search(r"step[_-]?(\d+)", file_info.relative_path)
                    if match:
                        return int(match.group(1))

                    # If no step found, try to extract any number from filename
                    match = re.search(r"(\d+)", file_info.name)
                    if match:
                        return int(match.group(1))

                    return 0
                except:
                    return 0

            # Organize images by folder and step
            folder_step_grid = {}
            all_steps = set()

            for file_info in media_files:
                # Get first level folder from relative path
                path_parts = Path(file_info.relative_path).parts
                if len(path_parts) > 1:  # Only include actual folders, not root files
                    folder_name = path_parts[0]

                    # Only include if folder is enabled
                    if folder_toggles.get(folder_name, True):
                        step_num = extract_step_number(file_info)
                        all_steps.add(step_num)

                        if folder_name not in folder_step_grid:
                            folder_step_grid[folder_name] = {}
                        folder_step_grid[folder_name][step_num] = file_info

            if not folder_step_grid:
                st.info(
                    "No experiments are selected to display. Use the sidebar toggles to select which experiments to show in the gallery.",
                    icon=":material/arrow_circle_left:",
                )
                return

            # Sort steps and folders
            sorted_steps = sorted(all_steps)
            sorted_folders = sorted(folder_step_grid.keys())

            # Calculate column-based navigation
            if layout_orientation == "Steps":
                # When steps are columns, paginate through steps
                total_columns = len(sorted_steps)
                column_items = sorted_steps
                column_type = "steps"
            else:
                # When experiments are columns, paginate through experiments
                total_columns = len(sorted_folders)
                column_items = sorted_folders
                column_type = "experiments"

            # Initialize current column index in session state
            if "current_column_index" not in st.session_state:
                st.session_state.current_column_index = 0

            # Pagination controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

            # Calculate current column range
            current_column_index = st.session_state.current_column_index
            current_columns = column_items[
                current_column_index : current_column_index + columns_per_page
            ]

            with col1:
                if st.button(
                    "First",
                    disabled=current_column_index == 0,
                    icon=":material/first_page:",
                    width="stretch",
                ):
                    st.session_state.current_column_index = 0
                    st.rerun()

            with col2:
                if st.button(
                    "Previous",
                    disabled=current_column_index == 0,
                    icon=":material/arrow_back:",
                    width="stretch",
                ):
                    # Move back by 1 column
                    st.session_state.current_column_index = max(0, current_column_index - 1)
                    st.rerun()

            with col3:
                if current_columns:
                    first_col = current_columns[0]
                    last_col = current_columns[-1]
                    st.write(
                        f"**{column_type.title()} {first_col} to {last_col}** ({total_columns} total {column_type})"
                    )
                else:
                    st.write(
                        f"**No {column_type} available** ({total_columns} total {column_type})"
                    )

            with col4:
                # Check if we can move forward by 1 column
                can_move_next = current_column_index + 1 + columns_per_page <= total_columns
                if st.button(
                    "Next",
                    disabled=not can_move_next,
                    icon=":material/arrow_forward:",
                    width="stretch",
                ):
                    # Move forward by 1 column
                    st.session_state.current_column_index = min(
                        total_columns - columns_per_page, current_column_index + 1
                    )
                    st.rerun()

            with col5:
                # Check if we're at the last possible position
                is_at_last = current_column_index + columns_per_page >= total_columns
                if st.button(
                    "Last", disabled=is_at_last, icon=":material/last_page:", width="stretch"
                ):
                    # Move to the last possible position
                    st.session_state.current_column_index = max(0, total_columns - columns_per_page)
                    st.rerun()

            # Get current columns based on column index
            current_columns = column_items[
                current_column_index : current_column_index + columns_per_page
            ]

            # Add column scrubber slider
            if column_items:
                # Create slider using actual column values
                current_first_col = current_columns[0] if current_columns else column_items[0]

                # Create slider for column selection using selectbox for discrete values
                selected_col = st.select_slider(
                    f"Jump to {column_type[:-1]}",
                    options=column_items,
                    value=current_first_col,
                    help=f"Use this slider to quickly jump to any {column_type[:-1]} in the series",
                )

                # Update current column index if slider value changed
                if selected_col in column_items:
                    new_index = column_items.index(selected_col)
                    if new_index != current_column_index:
                        st.session_state.current_column_index = new_index
                        st.rerun()

            if layout_orientation == "Steps":
                # Original layout: experiments as rows, steps as columns
                # Calculate optimal column widths with smart size limiting
                if sorted_folders:
                    # Find the longest experiment name
                    max_name_length = max(len(folder) for folder in sorted_folders)
                    # Add padding and convert to relative width (experiment names are typically 10-30 chars)
                    experiment_col_width = min(
                        max(max_name_length * 0.8, 12), 25
                    )  # Between 12 and 25

                    # Calculate available width for step columns
                    available_width = 100 - experiment_col_width

                    # Calculate step column width - each column can be smaller when more columns are added
                    step_col_width = available_width / len(current_columns)

                    # Smart size limiting: prevent any single image from being too large
                    # Reference size: 3 experiments, 4 files per page, but with smaller individual images
                    reference_experiment_width = 0  # Typical experiment column width
                    reference_available_width = 100 - reference_experiment_width
                    reference_step_width = reference_available_width / 4  # 4 files per page
                    # Reduce the maximum to 60% of the reference size for more reasonable single image size
                    max_step_width = (
                        reference_step_width * 1
                    )  # This is our maximum allowed step width

                    # Apply the size limit
                    if step_col_width > max_step_width:
                        step_col_width = max_step_width

                    # Create column configuration
                    col_config = [experiment_col_width] + [step_col_width] * len(current_columns)
                else:
                    col_config = [1] * (len(current_columns) + 1)

                # Create grid header with step numbers
                header_cols = st.columns(col_config)
                with header_cols[0]:
                    st.write("**Experiment**")
                for idx, step in enumerate(current_columns):
                    with header_cols[idx + 1]:
                        st.write(f"**Step {step}**")

                # Create grid rows (one per experiment)
                for folder_name in sorted_folders:
                    row_cols = st.columns(col_config)

                    # Experiment name in first column
                    with row_cols[0]:
                        st.write(folder_name)

                    # Images for each step in remaining columns
                    for idx, step in enumerate(current_columns):
                        with row_cols[idx + 1]:
                            if step in folder_step_grid[folder_name]:
                                file_info = folder_step_grid[folder_name][step]
                                try:
                                    # Check if it's a video file by extension
                                    file_extension = Path(file_info.path).suffix.lower()
                                    is_video = file_extension in _SUPPORTED_VIDEO_EXTENSIONS

                                    if is_video:
                                        # Display video using st.video
                                        st.video(file_info.path)
                                    else:
                                        # # Display image using PIL
                                        # image = Image.open(file_info.path)

                                        # # Apply consistent sizing if enabled
                                        # if consistent_size:
                                        #     image = image.resize(
                                        #         consistent_size, Image.Resampling.LANCZOS
                                        #     )

                                        # Display image
                                        st.image(file_info.path, width="stretch")
                                except Exception as e:
                                    st.error(f"Error loading {file_info.name}: {e}")
                            else:
                                st.write("—")  # No image for this step

            else:  # Experiments as Columns
                # New layout: steps as rows, experiments as columns
                # Calculate optimal column widths
                if sorted_folders:
                    # Find the longest experiment name
                    max_name_length = max(len(folder) for folder in sorted_folders)
                    # Add padding and convert to relative width
                    experiment_col_width = min(max(max_name_length * 0.8, 12), 25)

                    # Calculate available width for experiment columns
                    available_width = 100 - 15  # Reserve 15% for step labels

                    # Calculate experiment column width
                    experiment_col_width = available_width / len(current_columns)

                    # Apply smart size limiting for experiments too
                    max_experiment_width = 25  # Maximum 25% per experiment
                    if experiment_col_width > max_experiment_width:
                        experiment_col_width = max_experiment_width

                    # Create column configuration
                    col_config = [15] + [experiment_col_width] * len(current_columns)
                else:
                    col_config = [1] * (len(current_columns) + 1)

                # Create grid header with experiment names
                header_cols = st.columns(col_config)
                with header_cols[0]:
                    st.write("**Step**")
                for idx, folder_name in enumerate(current_columns):
                    with header_cols[idx + 1]:
                        st.write(f"**{folder_name}**")

                # Create grid rows (one per step)
                for step in sorted_steps:
                    row_cols = st.columns(col_config)

                    # Step number in first column
                    with row_cols[0]:
                        st.write(f"**{step}**")

                    # Images for each experiment in remaining columns
                    for idx, folder_name in enumerate(current_columns):
                        with row_cols[idx + 1]:
                            if step in folder_step_grid[folder_name]:
                                file_info = folder_step_grid[folder_name][step]
                                try:
                                    # Check if it's a video file by extension
                                    file_extension = Path(file_info.path).suffix.lower()
                                    is_video = file_extension in _SUPPORTED_VIDEO_EXTENSIONS

                                    if is_video:
                                        # Display video using st.video
                                        st.video(file_info.path)
                                    else:
                                        # Display image using PIL
                                        # image = Image.open(file_info.path)

                                        # # Apply consistent sizing if enabled
                                        # if consistent_size:
                                        #     image = Image.open(file_info.path).resize(
                                        #         consistent_size, Image.Resampling.LANCZOS
                                        #     )

                                        # Display image
                                        st.image(file_info.path, width="stretch")
                                except Exception as e:
                                    st.error(f"Error loading {file_info.name}: {e}")
                            else:
                                st.write("—")  # No image for this step

            # Show column range info
            if current_columns:
                st.info(f"Showing {column_type} {current_columns[0]} to {current_columns[-1]}")

        else:
            st.info("No media files found matching the current filters")

    else:
        st.warning("No media files found", icon=":material/info:")


if __name__ == "__main__":
    main()
