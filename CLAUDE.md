# Make Data Count - Project Guide

This project uses **uv** for Python dependency management and focuses on finding data references in academic papers using the Qwen3-0.6B local model.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- uv package manager installed

### Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

## 📁 Project Structure

```
MakeDataCount/
├── notebooks/
│   └── mdc-qwen3.ipynb     # Main analysis notebook
├── data/
│   ├── PDF/                # Place PDF files here
│   └── train_labels.csv    # Optional: for validation
├── pyproject.toml          # uv dependencies
├── CLAUDE.md              # This guide
└── .gitignore             # Git ignore rules
```

## 🔧 Common Commands

### Python Execution
```bash
# Run Python scripts with uv
uv run python script.py

# Run specific Python commands
uv run python -c "import torch; print(torch.cuda.is_available())"

# Install additional packages
uv add package-name

# Remove packages
uv remove package-name
```

### Jupyter Notebook
```bash
# Start Jupyter server
uv run jupyter notebook

# Or start JupyterLab
uv run jupyter lab

# Run notebook from command line
uv run jupyter nbconvert --execute --to notebook notebooks/mdc-qwen3.ipynb
```

### Model and Data Setup
```bash
# Create data directory
mkdir -p data/PDF

# Check GPU availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test vLLM installation
uv run python -c "import vllm; print('vLLM installed successfully')"
```

## 📊 Running the Analysis

### 1. Prepare Data
- Place PDF files in `data/PDF/` directory
- Optionally add `train_labels.csv` for validation

### 2. Execute Notebook
```bash
# Option 1: Interactive mode
uv run jupyter notebook notebooks/mdc-qwen3.ipynb

# Option 2: Command line execution
uv run jupyter nbconvert --execute --inplace notebooks/mdc-qwen3.ipynb
```

### 3. Expected Outputs
- `submission.csv`: Final results with dataset classifications
- Console output: Processing statistics and F1 scores

## 🛠 Development Workflow

### Code Quality
```bash
# Format code (if black is added)
uv run black .

# Type checking (if mypy is added)
uv run mypy .

# Run tests (if pytest is added)  
uv run pytest
```

### Dependency Management
```bash
# Show dependency tree
uv tree

# Update dependencies
uv sync --upgrade

# Export requirements
uv export --format requirements-txt > requirements.txt
```

## 🔍 Model Configuration

The project uses **Qwen2.5-0.5B-Instruct** model with these settings:
- Single GPU configuration
- 70% GPU memory utilization
- 4096 max context length
- Optimized for smaller model size

### Model Customization
To use a different model, edit `notebooks/mdc-qwen3.ipynb`:
```python
model_path = "your-model-name"  # Change this line
```

## 📋 Key Features

1. **DOI Extraction**: Identifies and extracts DOI references from academic texts
2. **Accession ID Detection**: Finds biological database identifiers
3. **Classification**: Categorizes data as Primary, Secondary, or None
4. **Reference Filtering**: Removes reference sections to focus on main content

## 🐛 Troubleshooting

### Common Issues

**vLLM V1 Logits Processor Error:**
```
ValueError: vLLM V1 does not support per request user provided logits processors
```
**Solution:** The notebook properly sets `VLLM_USE_V1="False"` before importing vllm. Restart kernel if error persists.

**CUDA Out of Memory:**
```python
# Reduce GPU memory utilization in notebook
gpu_memory_utilization=0.5  # Lower this value
```

**Model Download Fails:**
```bash
# Manually download model
uv run huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

**Dependency Conflicts:**
```bash
# Reset virtual environment
rm -rf .venv
uv sync
```

**PDF Processing Errors:**
- Ensure PDFs are not password protected
- Check file permissions in `data/PDF/` directory

### Performance Optimization

**For Limited GPU Memory:**
- Use smaller batch sizes
- Reduce `max_model_len` parameter
- Lower `gpu_memory_utilization`

**For Faster Processing:**
- Use SSD storage for model cache
- Increase batch sizes if memory allows
- Enable tensor parallelism for multi-GPU setups

## 📚 Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen Model Documentation](https://huggingface.co/Qwen)

## 🔄 Environment Management

```bash
# Show current environment
uv info

# Clean cache
uv cache clean

# Show installed packages
uv list

# Check for updates
uv sync --upgrade
```

## 📝 Notes for Claude Code

- Always use `uv run` prefix for Python commands
- Dependencies are locked in `uv.lock` file
- Virtual environment is automatically managed by uv
- Model files are cached in `.cache/` directory (gitignored)