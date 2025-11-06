# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-11-06

### Updated Dependencies

#### Core Libraries
- **Python**: Now requires >=3.10 (was >=3.8)
- **torch**: >=2.5.0 (was ==2.7.0 - fixed unrealistic version)
- **torchvision**: >=0.20.0 (was ==0.22.0)
- **transformers**: >=4.47.0 (was ==4.44.2)
- **tokenizers**: >=0.21.0 (was ==0.19.1)
- **sentencepiece**: >=0.2.0 (was ==0.1.99)
- **accelerate**: >=1.2.0 (was ==0.34.2)
- **peft**: >=0.14.0 (was unversioned)
- **bitsandbytes**: >=0.45.0 (was unversioned)
- **pydantic**: >=2.10.0 (was unversioned) - Major version upgrade to v2
- **gradio**: >=5.7.0 (was unversioned) - Major version upgrade to v5
- **numpy**: >=1.26.4,<2.1.0 (was ==1.26.4) - Added upper bound for compatibility
- **scikit-learn**: >=1.6.0 (was ==1.5.1)
- **deepspeed**: >=0.16.3 (was ==0.14.4)
- **wandb**: >=0.19.0 (was ==0.18.1)

#### New Dependencies
- **pillow**: >=11.0.0
- **protobuf**: >=5.29.0
- **huggingface-hub**: >=0.27.0

#### Third-party Libraries
- **opencole**: Updated langchain to 0.3.x (from 0.2.x)
- **skia-python**: >=87.5

### Fixed

#### Critical Bugs
- **Fixed logic error in `is_contain()` function** (`llava/metrics/layout.py:265`)
  - Changed `c3 = xr_2 >= xr_2` (always True) to `c3 = xr_1 >= xr_2`
  - This bug affected the underlay metrics calculation

#### Security Improvements
- **Removed default public sharing** in demo app
  - Changed `demo.launch(share=True)` to require explicit `--share` flag
  - Now launches locally by default (127.0.0.1:7860)
  - Added command-line arguments for better security control

#### Error Handling
- **Improved exception handling** across codebase
  - Changed bare `except:` to `except Exception as e:` with logging
  - Added warning messages for failed image loading
  - Better error messages help with debugging

### Added

#### New Constants
- Added `llava/constants.py` definitions:
  - `DEFAULT_IMAGE_SIZE = 336`
  - `DEFAULT_IMAGE_CHANNELS = 3`
  - `MAX_MODEL_LENGTH_SHORT = 5000`
  - `MAX_MODEL_LENGTH_LONG = 15000`
  - `LAYER_MAPPING` dictionary
  - `NUM_LAYERS = 5`

#### Command-Line Arguments (app.py)
- `--model_name_or_path`: Now required with help text
- `--share`: Opt-in flag for public URL sharing
- `--server-port`: Custom port configuration (default: 7860)
- `--server-name`: Custom host binding (default: 127.0.0.1)

#### Model Loading Improvements
- Added `trust_remote_code` parameter to `load_pretrained_model()`
- Added comprehensive docstring to model builder
- Better type hints and parameter documentation

### Changed

#### Code Quality
- **Replaced magic numbers with constants** throughout codebase
  - `torch.zeros(3, 336, 336)` → `torch.zeros(DEFAULT_IMAGE_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)`
  - Applied across `app/app.py`, `llava/infer/infer.py`, `llava/train/train.py`

#### Gradio API Updates (5.x compatibility)
- Updated `gr.Gallery` height parameters from strings to integers
  - `height="200px"` → `height=200`
- Added `type="filepath"` to `gr.File` component
- Updated event handling for Gradio 5.x API

#### Python Version
- Minimum Python version increased to 3.10
- Added Python 3.10, 3.11, 3.12 classifiers

#### Documentation
- Updated README.md with new demo launch options
- Added security warnings for `--share` flag
- Improved inline code documentation

### Deprecated
- Direct public sharing without explicit opt-in (security improvement)

### Removed
- Exact version pinning for most dependencies (allows easier updates)
- Hardcoded default port 5678 (now configurable, default 7860)

### API Compatibility Notes

#### Transformers 4.47.0+
- Added `trust_remote_code` parameter support
- Tokenizer loading remains compatible with fast/slow tokenizers
- All existing model architectures (LLaMA, Mistral, MPT) fully supported

#### Gradio 5.x
- File upload API updated but backward compatible
- Gallery components use integer heights
- Event handling updated to new API

#### Pydantic 2.x
- Major version upgrade from 1.x to 2.x
- Models should use Pydantic v2 API
- Existing schema definitions compatible

#### PyTorch 2.5+
- Flash Attention 2 support maintained
- BitsAndBytes quantization compatible
- DeepSpeed ZeRO stages 2/3 compatible

### Migration Guide

#### For Users
1. Update Python to 3.10 or higher
2. Reinstall dependencies: `pip install -e .`
3. Update demo launch commands (see README)
4. No model checkpoint changes required

#### For Developers
1. Import constants from `llava.constants`
2. Use `trust_remote_code` parameter if loading custom models
3. Update Gradio components to use integer heights
4. Replace bare `except:` with `except Exception as e:`

### Known Issues
- None at this time

### Testing
- All imports verified
- Model loading tested
- Gradio UI tested with new API
- Constants integration verified

---

## Previous Versions
No previous changelog entries available.
