# Quote Overlay: AI-Powered Quote Design Generator

Automatically generate aesthetically pleasing quote overlays on images using vision-language models.

![Quote Overlay Demo](assets/demo.png)

## 🎯 Overview

This project trains a LLaVA-based model to:
- Take a background image + quote text as input
- Predict optimal text placement, font, colors, and effects
- Generate professional-looking quote graphics

**Based on**: elem2design architecture (LLaVA + LoRA fine-tuning)

## 🚀 Quick Start (2 Weeks to Working Model)

### Week 1: Data Preparation
```bash
# 1. Download Crello dataset
python dataset/scripts/download_datasets.py

# 2. Filter for quote-style designs
python dataset/scripts/prepare_crello.py

# 3. Convert to training format
python dataset/scripts/create_dataset.py
```

### Week 2: Training
```bash
# Train model on Crello quote designs
python training/train_quote_model.py \
    --config training/training_config.yaml \
    --output_dir ./checkpoints/quote_v1
```

### Week 3: Demo
```bash
# Launch Gradio demo
python app/quote_app.py \
    --model_path ./checkpoints/quote_v1
```

## 📊 Datasets Used

### Primary Dataset: Crello
- **Source**: https://huggingface.co/datasets/cyberagent/crello
- **Size**: ~5,000-7,000 quote-style designs (filtered from 9,974 total)
- **Contains**: Professional designs with images + text overlays
- **License**: CDLA-Permissive-2.0

### Text Source: Quotes-500K
- **Source**: https://github.com/ShivaliGoel/Quotes-500K
- **Size**: 500,000 quotes
- **Categories**: Inspirational, motivational, life, love, philosophy

## 🏗️ Architecture

```
Input: Background Image (any size) + Quote Text
   ↓
Vision Encoder (CLIP ViT-L/14) - Frozen
   ↓
Multimodal Projector (MLP) - Frozen
   ↓
Language Model (Llama-3.2-3B) - LoRA Fine-tuned
   ↓
Output: JSON with design parameters
```

### Output Format
```json
{
  "text_box": {
    "left": 0.15,
    "top": 0.35,
    "width": 0.7,
    "height": 0.3
  },
  "text_style": {
    "font_family": "Playfair Display",
    "font_size": 48,
    "font_weight": "bold",
    "text_color": "#FFFFFF",
    "text_align": "center",
    "line_height": 1.4
  },
  "background_overlay": {
    "type": "solid_box",
    "color": "#000000",
    "opacity": 0.6,
    "padding": 20,
    "border_radius": 8
  },
  "effects": {
    "text_shadow": {
      "enabled": true,
      "color": "#000000",
      "blur": 4,
      "offset_x": 2,
      "offset_y": 2
    }
  }
}
```

## 📁 Project Structure

```
quote_overlay/
├── dataset/
│   ├── scripts/
│   │   ├── download_datasets.py      # Download Crello, quotes
│   │   ├── prepare_crello.py         # Filter quote designs
│   │   └── create_dataset.py         # Convert to training format
│   ├── config/
│   │   └── quote_config.yaml
│   └── data/                          # Downloaded data (gitignored)
├── training/
│   ├── train_quote_model.py          # Training script
│   └── training_config.yaml          # Hyperparameters
├── inference/
│   ├── generate_quote.py             # Generate design from image+text
│   └── render_quote.py               # Render final image
├── evaluation/
│   ├── evaluate_readability.py       # WCAG contrast, OCR accuracy
│   ├── evaluate_aesthetics.py        # Color harmony, composition
│   └── metrics.py                    # Metric implementations
├── app/
│   └── quote_app.py                  # Gradio web interface
├── checkpoints/                       # Model checkpoints (gitignored)
├── README.md
└── requirements.txt
```

## 📦 Installation

```bash
# Use elem2design environment
conda activate e2d

# Install additional dependencies
pip install -r quote_overlay/requirements.txt
```

## 🎓 Training Details

### Phase 1: Dataset Preparation
- Filter Crello for quote-style designs (~5k samples)
- Extract: background image, quote text, layout parameters
- Convert to LLaVA conversation format

### Phase 2: Model Training
- **Base Model**: meta-llama/Llama-3.2-3B
- **Vision Encoder**: CLIP ViT-L/14 (frozen)
- **Training Method**: LoRA (r=64, alpha=16)
- **Epochs**: 5
- **Batch Size**: 32 (8 per device × 4 accumulation steps)
- **Learning Rate**: 2e-4
- **Training Time**: 2-3 days on 4×A100 GPUs
- **Cost**: ~$1,500 (Lambda Labs)

### Phase 3: Evaluation
- **Readability Score**: Contrast ratio, OCR accuracy
- **Aesthetic Score**: Color harmony, compositional balance
- **Human Evaluation**: A/B testing vs baseline

## 📈 Expected Performance

| Metric | Target |
|--------|--------|
| Readability Score | 85-90% |
| Aesthetic Score | 80-85% |
| WCAG Contrast Compliance | 95%+ |
| Layout Validity | 98%+ |
| Human Preference (vs template) | 75-80% |

## 🎨 Usage

### Generate a Quote Overlay

```python
from inference.generate_quote import QuoteGenerator

# Initialize model
generator = QuoteGenerator(model_path="./checkpoints/quote_v1")

# Generate design
design = generator.generate(
    image_path="background.jpg",
    quote="The only way to do great work is to love what you do.",
    author="Steve Jobs"  # optional
)

# Render final image
from inference.render_quote import render_quote_image

output_image = render_quote_image(
    background_path="background.jpg",
    design_params=design,
    quote_text="The only way to do great work is to love what you do.",
    output_path="quote_output.jpg"
)
```

### Web Interface

```bash
python app/quote_app.py --model_path ./checkpoints/quote_v1
```

Then open http://127.0.0.1:7860 in your browser.

## 🔧 Configuration

Edit `training/training_config.yaml` to customize:
- Model architecture (base model, LoRA rank)
- Training hyperparameters (LR, epochs, batch size)
- Dataset filtering (minimum/maximum quote length)
- Output schema (which parameters to predict)

## 📊 Evaluation

```bash
# Run all evaluations
python evaluation/evaluate_readability.py --model_path ./checkpoints/quote_v1
python evaluation/evaluate_aesthetics.py --model_path ./checkpoints/quote_v1

# Generate evaluation report
python evaluation/generate_report.py --checkpoint ./checkpoints/quote_v1
```

## 🚀 Deployment

### Option 1: Local Inference
```bash
python inference/generate_quote.py \
    --model_path ./checkpoints/quote_v1 \
    --image background.jpg \
    --quote "Your quote here"
```

### Option 2: API Server
```bash
# Start FastAPI server
python app/api_server.py --model_path ./checkpoints/quote_v1 --port 8000

# Use API
curl -X POST http://localhost:8000/generate \
  -F "image=@background.jpg" \
  -F "quote=Your quote here"
```

### Option 3: Gradio Demo
```bash
python app/quote_app.py --model_path ./checkpoints/quote_v1
```

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| Datasets | $0 (open-source) |
| Training (5 days, 4×A100) | $1,500 |
| Inference GPU (T4) | $250/month |
| Storage (S3) | $20/month |
| **Total Initial** | **$1,770** |
| **Monthly Running** | **$270** |

## 🎯 Roadmap

### v1.0 (Current - Week 1-3)
- [x] Project setup
- [ ] Download and filter Crello dataset
- [ ] Train on Crello quote designs
- [ ] Basic Gradio demo

### v1.1 (Week 4-6)
- [ ] Add PKU PosterLayout dataset
- [ ] Improve color harmony selection
- [ ] Multi-font support (20+ fonts)
- [ ] Better readability metrics

### v2.0 (Month 2-3)
- [ ] Generate synthetic training data
- [ ] RLHF for aesthetic refinement
- [ ] Style transfer (match reference designs)
- [ ] Animation parameters

### v3.0 (Future)
- [ ] Multi-language support
- [ ] Video quote generation
- [ ] API service with authentication
- [ ] Mobile app integration

## 📝 Citation

If you use this project, please cite:

```bibtex
@misc{quote_overlay_2025,
  title={Quote Overlay: AI-Powered Quote Design Generator},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/elem2design}}
}
```

## 📄 License

This project builds upon elem2design and inherits its license terms.

## 🙏 Acknowledgments

- **elem2design**: Base architecture and training pipeline
- **Crello Dataset**: CyberAgent for the design dataset
- **LLaVA**: Haotian Liu et al. for the vision-language model
- **PKU PosterLayout**: Peking University for the layout benchmark

## 🐛 Issues & Support

For questions or issues, please open a GitHub issue or contact [your email].

## 📚 Additional Resources

- [elem2design Paper](https://arxiv.org/abs/2412.19712)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Crello Dataset](https://huggingface.co/datasets/cyberagent/crello)
- [PKU PosterLayout](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023)
