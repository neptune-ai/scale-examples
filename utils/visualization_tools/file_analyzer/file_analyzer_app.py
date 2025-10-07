import streamlit as st
import os
import pandas as pd
from pathlib import Path
import base64
from PIL import Image
import io
import json
import re
# import matplotlib.pyplot as plt

# import seaborn as sns
from typing import List, Dict, Any
# import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

# Neptune integration
try:
    import neptune_query as nq
    import neptune_query.runs as nq_runs
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="File Series Visualizer",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_file_icon(file_path: str) -> str:
    """Get appropriate icon for file type"""
    ext = Path(file_path).suffix.lower()
    icon_map = {
        '.py': '🐍',
        '.ipynb': '📓',
        '.json': '📄',
        '.csv': '📊',
        '.txt': '📝',
        '.md': '📖',
        '.png': '🖼️',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.gif': '🖼️',
        '.mp4': '🎥',
        '.mp3': '🎵',
        '.wav': '🎵',
        '.pdf': '📕',
        '.zip': '📦',
        '.tar': '📦',
        '.gz': '📦'
    }
    return icon_map.get(ext, '📄')

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def is_media_file(file_path: str) -> bool:
    """Check if file is a static image, gif, or video"""
    media_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.webm'}
    return Path(file_path).suffix.lower() in media_extensions

def is_video_file(file_path: str) -> bool:
    """Check if file is a video"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    return Path(file_path).suffix.lower() in video_extensions

def is_text_file(file_path: str) -> bool:
    """Check if file is a text file"""
    text_extensions = {'.txt', '.py', '.json', '.csv', '.md', '.yaml', '.yml', '.xml', '.html', '.css', '.js'}
    return Path(file_path).suffix.lower() in text_extensions

def scan_directory(directory: str, recursive: bool = True) -> List[Dict[str, Any]]:
    """Scan directory and return file information"""
    files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        return files
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in directory_path.glob(pattern):
        if file_path.is_file():
            try:
                file_info = {
                    'name': file_path.name,
                    'path': str(file_path),
                    'relative_path': str(file_path.relative_to(directory_path)),
                    'size_mb': get_file_size_mb(str(file_path)),
                    'extension': file_path.suffix.lower(),
                    'icon': get_file_icon(str(file_path)),
                    'is_media': is_media_file(str(file_path)),
                    'is_text': is_text_file(str(file_path)),
                    'modified': file_path.stat().st_mtime
                }
                files.append(file_info)
            except Exception as e:
                st.warning(f"Error reading {file_path}: {e}")
    
    return files

def display_image_preview(file_path: str, max_width: int = 300, consistent_size: tuple = None):
    """Display image preview with optional consistent sizing"""
    try:
        image = Image.open(file_path)
        
        # Apply consistent sizing if specified
        if consistent_size:
            image = image.resize(consistent_size, Image.Resampling.LANCZOS)
        else:
            # Resize if too large
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        st.image(image, caption=Path(file_path).name, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

def display_text_preview(file_path: str, max_lines: int = 50):
    """Display text file preview"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            st.text_area(
                f"Preview of {Path(file_path).name} (showing first {max_lines} lines)",
                ''.join(lines[:max_lines]),
                height=300,
                disabled=True
            )
            st.info(f"File has {len(lines)} lines total. Showing first {max_lines} lines.")
        else:
            st.text_area(
                f"Content of {Path(file_path).name}",
                ''.join(lines),
                height=300,
                disabled=True
            )
    except Exception as e:
        st.error(f"Error reading text file: {e}")

def create_file_statistics(files: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create file statistics DataFrame"""
    if not files:
        return pd.DataFrame()
    
    df = pd.DataFrame(files)
    df['modified_date'] = pd.to_datetime(df['modified'], unit='s')
    
    return df

@st.cache_data()
def download_neptune_files(project_name: str, experiment_regex: str, attribute_regex: str, download_dir: str) -> List[Dict[str, Any]]:
    """Download files from Neptune and return file information"""
    if not NEPTUNE_AVAILABLE:
        st.error("Neptune Query is not available. Please install neptune-query package.")
        return []
    
    try:
        # List experiments
        exps = nq.list_experiments(project=project_name, 
                                   experiments=f"{experiment_regex}")
        
        # Filter experiments by regex
        filtered_exps = []
        for exp in exps:
            if re.match(experiment_regex, exp):
                filtered_exps.append(exp)
        
        if not filtered_exps:
            st.warning(f"No experiments found matching pattern: {experiment_regex}")
            return []
        
        # Fetch files from experiments using the attribute regex
        files = nq.fetch_series(
            project=project_name,
            experiments=filtered_exps,
            attributes=attribute_regex
        )
        
        # Create project-specific download directory
        # Use project name as top-level folder to prevent mixing experiments from different projects
        project_download_dir = Path(download_dir) / project_name.replace("/", "_")
        project_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download files to project-specific directory
        nq.download_files(files=files, destination=str(project_download_dir))
        
        # Convert to our file format
        downloaded_files = []
        
        # Scan the project-specific download directory for files
        all_files = list(project_download_dir.rglob("*"))
        
        media_count = 0
        for file_path in all_files:
            if file_path.is_file():
                try:
                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'relative_path': str(file_path.relative_to(project_download_dir)),
                        'size_mb': get_file_size_mb(str(file_path)),
                        'extension': file_path.suffix.lower(),
                        'icon': get_file_icon(str(file_path)),
                        'is_media': is_media_file(str(file_path)),
                        'is_video': is_video_file(str(file_path)),
                        'is_text': is_text_file(str(file_path)),
                        'modified': file_path.stat().st_mtime
                    }
                    downloaded_files.append(file_info)
                    
                    if file_info['is_media']:
                        media_count += 1
                        
                except Exception as e:
                    st.warning(f"Error processing file {file_path}: {e}")
        
        # Store download info for display in expander
        download_info = {
            'project_name': project_name,
            'experiments': filtered_exps,
            'attribute_regex': attribute_regex,
            'files_fetched': len(files),
            'download_dir': str(project_download_dir),
            'total_files': len(all_files),
            'media_files': media_count,
            'total_processed': len(downloaded_files)
        }
        
        return downloaded_files, download_info
        
    except Exception as e:
        st.error(f"Error downloading from Neptune: {e}")
        return [], {}

def main():
    st.title("📁 File Series Visualizer")
    st.markdown("Visualize and explore files across different folders")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Project configuration
    st.sidebar.subheader("📂 Project Configuration")
    
    if not NEPTUNE_AVAILABLE:
        st.sidebar.error("Neptune Query not available. Install with: pip install neptune-query")
    else:
        # Neptune project
        neptune_project = st.sidebar.text_input(
            "Neptune Project",
            value=st.session_state.get('neptune_project'),
            help="Neptune project name (e.g., 'examples/quickstart')"
        )
        st.session_state.neptune_project = neptune_project
        
        # Download directory
        download_directory = st.sidebar.text_input(
            "Download Directory",
            value=st.session_state.get('download_directory', os.path.join(os.getcwd(), 'neptune_downloads')),
            help="Directory to download Neptune files to"
        )
        st.session_state.download_directory = download_directory
        
        # Experiment regex
        experiment_regex = st.sidebar.text_input(
            "Experiment Regex",
            value=st.session_state.get('experiment_regex', '.*'),
            help="Regex pattern to match experiment names (e.g., 'exp_.*' or '.*_v[0-9]+')"
        )
        st.session_state.experiment_regex = experiment_regex
        
        # Attribute regex
        attribute_regex = st.sidebar.text_input(
            "Attribute Regex",
            value=st.session_state.get('attribute_regex', '.*'),
            help="Regex pattern to match file attributes/names (e.g., 'step_.*' or '.*_result.*')"
        )
        st.session_state.attribute_regex = attribute_regex
        
        if st.sidebar.button("⬇️ Fetch and Visualize"):
            with st.spinner("Downloading files from Neptune..."):
                files, download_info = download_neptune_files(neptune_project, experiment_regex, attribute_regex, download_directory)
                st.session_state.files = files
                st.session_state.download_info = download_info
                st.session_state.directory_scanned = True
                
                # Show success/warning message
                if files:
                    st.success(f"Successfully downloaded {len(files)} files")
                else:
                    st.warning("No files were downloaded. Check your project name and regex patterns.")
        
        # Show download details in expander if available
        if 'download_info' in st.session_state and st.session_state.download_info:
            with st.sidebar.expander("📋 Download Details", expanded=False):
                info = st.session_state.download_info
                st.write(f"**Project:** {info.get('project_name', 'N/A')}")
                st.write(f"**Experiments Found:** {len(info['experiments'])}")
                st.write(f"**Experiments:** {', '.join(info['experiments'])}")
                st.write(f"**Attribute Regex:** {info['attribute_regex']}")
                st.write(f"**Files Fetched:** {info['files_fetched']}")
                st.write(f"**Download Directory:** {info['download_dir']}")
                st.write(f"**Total Files:** {info['total_files']}")
                st.write(f"**Media Files:** {info['media_files']}")
                st.write(f"**Files Processed:** {info['total_processed']}")
    
    # Gallery view options
    if 'files' in st.session_state and st.session_state.files:
        st.sidebar.subheader("👁️ Gallery Options")
        
        # Apply regex filters to get experiments and media files
        import re
        
        experiment_pattern = st.session_state.get('experiment_regex', '.*')
        attribute_pattern = st.session_state.get('attribute_regex', '.*')
        
        # Filter files by media type and regex patterns
        filtered_files = []
        for file_info in st.session_state.files:
            # Check if it's a media file
            if file_info.get('is_media', False):
                # Check experiment regex (folder name)
                path_parts = Path(file_info['relative_path']).parts
                if len(path_parts) > 1:
                    experiment_name = path_parts[0]
                    if re.match(experiment_pattern, experiment_name):
                        # Check attribute regex (filename)
                        if re.match(attribute_pattern, file_info['name']):
                            filtered_files.append(file_info)
        
        # Get unique experiments from filtered files
        top_level_folders = set()
        for file_info in filtered_files:
            path_parts = Path(file_info['relative_path']).parts
            if len(path_parts) > 1:
                top_level_folders.add(path_parts[0])
        
        # Create individual toggles for each experiment
        folder_toggles = {}
        if top_level_folders:
            st.sidebar.write("**Select experiments to show in gallery:**")
            for folder in sorted(top_level_folders):
                folder_toggles[folder] = st.sidebar.checkbox(
                    folder, 
                    value=True,  # Default to showing all folders
                    key=f"folder_toggle_{folder}"
                )
        
        # Gallery layout controls
        st.sidebar.subheader("📄 Gallery Layout")
        
        # Layout orientation toggle
        layout_orientation = st.sidebar.radio(
            "Layout Orientation",
            ["Steps as Columns", "Steps as Rows"],
            help="Choose how to arrange steps in the gallery"
        )
        
        # Pagination controls
        images_per_page = st.sidebar.slider(
            "Media files per page", 
            min_value=1, 
            max_value=10, 
            value=5, 
            help="Number of media files to show at once"
        )
        
        # Image sizing controls
        st.sidebar.subheader("🖼️ Image Display")
        consistent_sizing = st.sidebar.checkbox(
            "Consistent image size",
            value=True,
            help="Resize all images to the same dimensions for easier comparison"
        )
        
        if consistent_sizing:
            image_width = st.sidebar.slider(
                "Image width (pixels)",
                min_value=100,
                max_value=500,
                value=200,
                step=10
            )
            image_height = st.sidebar.slider(
                "Image height (pixels)",
                min_value=100,
                max_value=500,
                value=200,
                step=10
            )
            consistent_size = (image_width, image_height)
        else:
            consistent_size = None
        
        
        st.session_state.filtered_files = filtered_files
        st.session_state.folder_toggles = folder_toggles
        st.session_state.images_per_page = images_per_page
        st.session_state.consistent_size = consistent_size
        st.session_state.layout_orientation = layout_orientation
    
    # Main content area
    if 'files' not in st.session_state or not st.session_state.files:
        st.info("👆 Please configure your Neptune project and click 'Fetch and Visualize' to get started")
        return
    
    # Image comparison gallery
    
    filtered_df = create_file_statistics(st.session_state.get('filtered_files', st.session_state.files))
    
    if not filtered_df.empty:
        # Gallery view only
        view_mode = "Gallery"
        
        if view_mode == "Gallery":
            # Grid gallery: rows = experiments, columns = steps
            media_files = [f for f in filtered_df.itertuples() if f.is_media]
            
            if media_files:
                st.subheader("Comparison Grid")
                
                # Get folder toggles, pagination settings, and image sizing
                folder_toggles = st.session_state.get('folder_toggles', {})
                images_per_page = st.session_state.get('images_per_page', 3)
                consistent_size = st.session_state.get('consistent_size', None)
                layout_orientation = st.session_state.get('layout_orientation', 'Steps as Columns')
                
                # Extract step number from filename or path
                def extract_step_number(file_info):
                    try:
                        import re
                        # Try to extract step from filename first
                        match = re.search(r'step_(\d+)', file_info.name)
                        if match:
                            return int(match.group(1))
                        
                        # Try to extract step from path (for Neptune downloads)
                        match = re.search(r'step[_-]?(\d+)', file_info.path)
                        if match:
                            return int(match.group(1))
                        
                        # Try to extract step from relative path
                        match = re.search(r'step[_-]?(\d+)', file_info.relative_path)
                        if match:
                            return int(match.group(1))
                        
                        # If no step found, try to extract any number from filename
                        match = re.search(r'(\d+)', file_info.name)
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
                    st.info("No experiments are selected to display. Use the sidebar toggles to select which experiments to show in the gallery.")
                    return
                
                # Sort steps and folders
                sorted_steps = sorted(all_steps)
                sorted_folders = sorted(folder_step_grid.keys())
                
                # Calculate step-based navigation
                total_steps = len(sorted_steps)
                
                # Initialize current step index in session state
                if 'current_step_index' not in st.session_state:
                    st.session_state.current_step_index = 0
                
                # Pagination controls
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                # Calculate current step range
                current_step_index = st.session_state.current_step_index
                current_steps = sorted_steps[current_step_index:current_step_index + images_per_page]
                
                with col1:
                    if st.button("⏮️ First", disabled=current_step_index == 0):
                        st.session_state.current_step_index = 0
                        st.rerun()
                
                with col2:
                    if st.button("⬅️ Previous", disabled=current_step_index == 0):
                        # Move back by 1 step
                        st.session_state.current_step_index = max(0, current_step_index - 1)
                        st.rerun()
                
                with col3:
                    if current_steps:
                        first_step = current_steps[0]
                        last_step = current_steps[-1]
                        st.write(f"**Steps {first_step} to {last_step}** ({total_steps} total steps)")
                    else:
                        st.write(f"**No steps available** ({total_steps} total steps)")
                
                with col4:
                    # Check if we can move forward by 1 step
                    can_move_next = current_step_index + 1 + images_per_page <= total_steps
                    if st.button("➡️ Next", disabled=not can_move_next):
                        # Move forward by 1 step
                        st.session_state.current_step_index = min(total_steps - images_per_page, current_step_index + 1)
                        st.rerun()
                
                with col5:
                    # Check if we're at the last possible position
                    is_at_last = current_step_index + images_per_page >= total_steps
                    if st.button("⏭️ Last", disabled=is_at_last):
                        # Move to the last possible position
                        st.session_state.current_step_index = max(0, total_steps - images_per_page)
                        st.rerun()
                
                # Get current steps based on step index
                current_steps = sorted_steps[current_step_index:current_step_index + images_per_page]
                
                # Add step scrubber slider
                if sorted_steps:
                    st.subheader("Step Scrubber")
                    
                    # Create slider using actual step values
                    current_first_step = current_steps[0] if current_steps else sorted_steps[0]
                    current_step_index_in_list = sorted_steps.index(current_first_step) if current_first_step in sorted_steps else 0
                    
                    # Create slider for step selection using selectbox for discrete values
                    selected_step = st.select_slider(
                        "Jump to step:",
                        options=sorted_steps,
                        value=current_first_step,
                        help="Use this slider to quickly jump to any step in the series"
                    )
                    
                    # Update current step index if slider value changed
                    if selected_step in sorted_steps:
                        new_index = sorted_steps.index(selected_step)
                        if new_index != current_step_index:
                            st.session_state.current_step_index = new_index
                            st.rerun()
                
                if layout_orientation == "Steps as Columns":
                    # Original layout: experiments as rows, steps as columns
                    # Calculate optimal column widths with smart size limiting
                    if sorted_folders:
                        # Find the longest experiment name
                        max_name_length = max(len(folder) for folder in sorted_folders)
                        # Add padding and convert to relative width (experiment names are typically 10-30 chars)
                        experiment_col_width = min(max(max_name_length * 0.8, 12), 25)  # Between 12 and 25
                        
                        # Calculate available width for step columns
                        available_width = 100 - experiment_col_width
                        
                        # Calculate step column width - each column can be smaller when more columns are added
                        step_col_width = available_width / images_per_page
                        
                        # Smart size limiting: prevent any single image from being too large
                        # Reference size: 3 experiments, 4 files per page, but with smaller individual images
                        reference_experiment_width = 20  # Typical experiment column width
                        reference_available_width = 100 - reference_experiment_width
                        reference_step_width = reference_available_width / 4  # 4 files per page
                        # Reduce the maximum to 60% of the reference size for more reasonable single image size
                        max_step_width = reference_step_width * 0.6  # This is our maximum allowed step width
                        
                        # Apply the size limit
                        if step_col_width > max_step_width:
                            step_col_width = max_step_width
                        
                        # Create column configuration
                        col_config = [experiment_col_width] + [step_col_width] * images_per_page
                    else:
                        col_config = [1] * (images_per_page + 1)
                    
                    # Create grid header with step numbers
                    header_cols = st.columns(col_config)
                    with header_cols[0]:
                        st.write("**Experiment**")
                    for idx, step in enumerate(current_steps):
                        with header_cols[idx + 1]:
                            st.write(f"**Step {step}**")
                    
                    # Create grid rows (one per experiment)
                    for folder_name in sorted_folders:
                        row_cols = st.columns(col_config)
                        
                        # Experiment name in first column
                        with row_cols[0]:
                            st.write(f"👁️ {folder_name}")
                        
                        # Images for each step in remaining columns
                        for idx, step in enumerate(current_steps):
                            with row_cols[idx + 1]:
                                if step in folder_step_grid[folder_name]:
                                    file_info = folder_step_grid[folder_name][step]
                                    try:
                                        # Check if it's a video file by extension
                                        file_extension = Path(file_info.path).suffix.lower()
                                        is_video = file_extension in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                                        
                                        if is_video:
                                            # Display video using st.video
                                            st.video(file_info.path)
                                        else:
                                            # Display image using PIL
                                            image = Image.open(file_info.path)
                                            
                                            # Apply consistent sizing if enabled
                                            if consistent_size:
                                                image = image.resize(consistent_size, Image.Resampling.LANCZOS)
                                            
                                            # Display image
                                            st.image(image, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error loading {file_info.name}: {e}")
                                else:
                                    st.write("—")  # No image for this step
                
                else:  # Steps as Rows
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
                        experiment_col_width = available_width / len(sorted_folders)
                        
                        # Apply smart size limiting for experiments too
                        max_experiment_width = 25  # Maximum 25% per experiment
                        if experiment_col_width > max_experiment_width:
                            experiment_col_width = max_experiment_width
                        
                        # Create column configuration
                        col_config = [15] + [experiment_col_width] * len(sorted_folders)
                    else:
                        col_config = [1] * (len(sorted_folders) + 1)
                    
                    # Create grid header with experiment names
                    header_cols = st.columns(col_config)
                    with header_cols[0]:
                        st.write("**Step**")
                    for idx, folder_name in enumerate(sorted_folders):
                        with header_cols[idx + 1]:
                            st.write(f"**👁️ {folder_name}**")
                    
                    # Create grid rows (one per step)
                    for step in current_steps:
                        row_cols = st.columns(col_config)
                        
                        # Step number in first column
                        with row_cols[0]:
                            st.write(f"**Step {step}**")
                        
                        # Images for each experiment in remaining columns
                        for idx, folder_name in enumerate(sorted_folders):
                            with row_cols[idx + 1]:
                                if step in folder_step_grid[folder_name]:
                                    file_info = folder_step_grid[folder_name][step]
                                    try:
                                        # Check if it's a video file by extension
                                        file_extension = Path(file_info.path).suffix.lower()
                                        is_video = file_extension in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                                        
                                        if is_video:
                                            # Display video using st.video
                                            st.video(file_info.path)
                                        else:
                                            # Display image using PIL
                                            image = Image.open(file_info.path)
                                            
                                            # Apply consistent sizing if enabled
                                            if consistent_size:
                                                image = image.resize(consistent_size, Image.Resampling.LANCZOS)
                                            
                                            # Display image
                                            st.image(image, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error loading {file_info.name}: {e}")
                                else:
                                    st.write("—")  # No image for this step
                
                # Show step range info
                if current_steps:
                    st.info(f"Showing steps {current_steps[0]} to {current_steps[-1]}")
                
            else:
                st.info("No media files found matching the current filters")
    
    else:
        st.warning("No files match the current filters")

if __name__ == "__main__":
    main()
