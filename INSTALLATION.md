# Installation Guide

## Install as Editable Package

This is the **recommended** way to work with this project during development.

### Step 1: Create a virtual environment (recommended)

```bash
cd /home/mrm367/Multi-Hop-Reasoning
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR on Windows: venv\Scripts\activate
```

### Step 2: Install in editable mode

```bash
pip install -e .
```

This will:
- Install all dependencies from `requirements.txt`
- Make `eval` and `hotpotqa` modules importable from anywhere
- Allow you to edit code and see changes immediately (no reinstall needed)
- Create command-line tools: `hotpotqa-evaluate` and `hotpotqa-predict`

### Step 3: Verify installation

```bash
python -c "import hotpotqa; print('✓ hotpotqa module available')"
python -c "import eval.hotpotqa; print('✓ eval module available')"
```

Or run the test script:

```bash
python scripts/test_eval_vs_hotpotqa.py
```

## Using the Installed Package

### In Python scripts

```python
# Clean imports - no sys.path manipulation needed!
from hotpotqa import load_hotpotqa_dataset, evaluate_predictions
from eval.hotpotqa import evaluate_hotpotqa
```

### Command-line tools

After installation, you can use these commands from anywhere:

```bash
# Evaluate predictions
hotpotqa-evaluate --predictions my_predictions.json --split dev

# Generate predictions
hotpotqa-predict --strategy oracle --split dev --output predictions.json
```

## Development Installation (Optional)

If you want development tools (pytest, black, ruff):

```bash
pip install -e ".[dev]"
```

## Alternative: Regular Installation

If you don't need to edit the code:

```bash
pip install .
```

Note: With regular installation, you'll need to reinstall after code changes.

## Troubleshooting

### ImportError after installation
- Make sure you're in the activated virtual environment
- Try: `pip install -e . --force-reinstall`

### Scripts can't find modules
- Check: `pip list | grep multi-hop`
- Should show: `multi-hop-reasoning` with your project path

### OPENAI_API_KEY not found
```bash
export OPENAI_API_KEY="your-key-here"
```

