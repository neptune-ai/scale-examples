# Neptune Experiment Comparison Tool

A Streamlit application for downloading and comparing media files across Neptune experiments with step-based navigation and interactive visualization.

## Features

- **Neptune Integration**: Download files directly from your Neptune experiments
- **Experiment Filtering**: Use regex patterns to filter experiments and file attributes
- **Step-Based Comparison**: Compare images/videos across experiments at specific training steps
- **Interactive Grid View**:
  - Rows represent experiments
  - Columns represent training steps
  - Easy side-by-side comparison
- **Media File Support**:
  - Static images (PNG, JPG, JPEG, BMP, TIFF, WebP)
  - Animated images (GIF)
  - Videos (MP4, AVI, MOV, MKV, WebM)
- **Smart Navigation**:
  - Previous/Next buttons for sequential browsing
  - Step scrubber slider for quick jumping
  - Pagination controls for viewing multiple steps
- **Consistent Sizing**: Optional uniform image dimensions for better comparison

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run file_analyzer_app.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Configure your Neptune project:
   - **Neptune Project**: Enter your Neptune project name (e.g., "examples/LLM-Pretraining")
   - **Download Directory**: Choose where to save downloaded files
   - **Experiment Regex**: Pattern to match experiment names (e.g., "llm_train-v.*")
   - **Attribute Regex**: Pattern to match file attributes (e.g., "eval/attention_maps")

4. Click "Download & Visualize" to fetch and display your experiment files

5. Use the comparison grid to:
   - Toggle experiments on/off using the sidebar checkboxes
   - Navigate through steps using Previous/Next buttons
   - Jump to specific steps using the step scrubber slider
   - Adjust image sizing and pagination settings

## Example Workflows

### Training Progress Analysis
1. Set experiment regex to match your training runs (e.g., "exp_.*")
2. Set attribute regex to target specific outputs (e.g., "generated_images")
3. Use step scrubber to navigate through training progression
4. Compare image quality across different experiments

### A/B Testing
1. Configure experiments to compare different model versions
2. Toggle experiments on/off to focus on specific comparisons
3. Use consistent sizing to eliminate visual bias
4. Navigate through steps to see performance differences

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Pillow (PIL)
- Neptune Query (neptune-query)

## Dependencies

All required packages are listed in `requirements.txt`

## Changelog

### Version 0.1.0 - Neptune Experiment File Analyzer
**Initial Release - Specialized tool for comparing files logged to Neptune**

#### Features
- **Neptune Integration**: Direct download from Neptune
- **Experiment Filtering**: Regex-based filtering for experiments and attributes
- **Step-Based Comparison Grid**: Rows=experiments, Columns=training steps
- **Smart Navigation**:
  - Previous/Next buttons with sliding window navigation
  - Step scrubber slider for quick jumping to specific steps
  - Pagination controls (1-10 steps per view)
- **Media File Support**: Specialized support for images, and videos
- **Consistent Image Sizing**: Optional uniform dimensions for better comparison
- **Experiment Toggles**: Individual checkboxes to show/hide experiments
- **Step Extraction**: Automatic step number detection from file paths
- **Quality Resizing**: High-quality image resampling with customizable dimensions
