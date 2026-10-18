# Python 3.14 Update

This project has been updated to support Python 3.14.

## Changes Made

### Updated Python Version Requirement

- **Previous**: Python 3.13
- **Current**: Python 3.14

### Updated Dependencies

The following dependencies have been updated for Python 3.14 compatibility:

| Package | Previous Version | New Version | Notes |
|---------|-----------------|-------------|-------|
| numpy | 2.2.6 | >=2.5.0 | NumPy 2.5+ adds Python 3.14 support |
| torch | 2.7.1 | >=2.10.0 | PyTorch 2.10+ adds Python 3.14 support |

### Files Modified

- **`.python-version`** - Added to specify Python 3.14
- **`README.md`** - Updated setup instructions to mention Python 3.14
- **`requirements.txt`** - Updated version constraints for numpy and torch
- **`requirements-linux-cpu.txt`** - Updated version constraints for numpy and torch

## System Environment

- **Operating System**: Windows 11
- **Shell**: PowerShell
- **Python Version**: 3.14.4

## Setting Up with Python 3.14

### Prerequisites

- Python 3.14 installed on your system
- Verify with: `python --version`

### Installation Steps

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Or using `uv`:

```powershell
uv venv
uv pip install -r requirements.txt
```

**Note for Linux users**: Use `requirements-linux-cpu.txt` instead:
```powershell
pip install -r requirements-linux-cpu.txt --index-strategy unsafe-best-match
```

## Testing

After installation, run the workshop steps (all commands executed via PowerShell):

```powershell
python step01-prepare-data.py
python step02-build-model.py
python step03-train-model.py
python step04-generate-text.py
```

## Notes

- If you encounter issues with `pip`, ensure you're using Python 3.14's pip: `python -m pip`
- The project uses CPU-only builds of PyTorch (no CUDA required)
- All workshop functionality remains unchanged - only the Python and dependency versions were updated
