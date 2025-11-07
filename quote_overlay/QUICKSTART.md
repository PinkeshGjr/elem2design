# Quick Start Guide: Quote Overlay Project

This guide will help you get started with training a quote overlay model in just 2 weeks.

## 📋 Prerequisites

- GPU with at least 24GB VRAM (for training) or 8GB (for inference)
- 200GB free disk space
- Python 3.10+
- elem2design environment set up

## 🚀 Week 1: Data Preparation

### Day 1: Download Datasets

```bash
cd quote_overlay

# Download Crello dataset and quotes
python dataset/scripts/download_datasets.py --output_dir ./data

# This will download:
# - Crello dataset (~50GB)
# - Quotes-500K dataset (~50MB)
```

**Expected time**: 2-4 hours (depending on internet speed)

### Day 2-3: Filter Quote-Style Designs

```bash
# Filter Crello for quote-style designs
python dataset/scripts/prepare_crello.py \
    --crello_path ./data/crello \
    --output_dir ./data/filtered \
    --min_quote_length 10 \
    --max_quote_length 200

# This will:
# - Analyze all Crello designs
# - Filter for quote-style layouts (simple, text-focused)
# - Save ~5,000-7,000 filtered samples
```

**Expected output**: 5,000-7,000 quote designs

### Day 4-5: Convert to Training Format

```bash
# Convert filtered data to LLaVA conversation format
python dataset/scripts/create_dataset.py \
    --filtered_dir ./data/filtered \
    --output_dir ./data/llava_format \
    --image_folder ./data/crello_images

# This creates:
# - train.json (~4,000 samples)
# - validation.json (~500 samples)
# - test.json (~500 samples)
```

**Checkpoint**: Verify you have:
- ✓ `data/llava_format/train.json`
- ✓ `data/llava_format/validation.json`
- ✓ `data/llava_format/test.json`

---

## 🎓 Week 2: Training

### Day 6-10: Model Training

```bash
# Start training
python training/train_quote_model.py \
    --config training/training_config.yaml

# Training will run for 5 epochs (~2-3 days on 4×A100 GPUs)

# Monitor training:
# - Check logs in ./checkpoints/quote_v1/
# - Best model saved at ./checkpoints/quote_v1/checkpoint-XXXX/
```

**Hardware options**:
- **Cloud GPU** (Lambda Labs, AWS, GCP)
  - 4× A100 GPUs: $3.20/hour × 60 hours = $192
- **Consumer GPU** (RTX 4090, A6000)
  - Single GPU: Slower but cheaper (~7-10 days)

**Training progress**:
- Epoch 1: Initial learning
- Epoch 2-3: Rapid improvement
- Epoch 4-5: Fine-tuning and convergence

### Day 11: Evaluation

```bash
# Evaluate on test set
python evaluation/evaluate_readability.py \
    --model_path ./checkpoints/quote_v1/checkpoint-best

# Check metrics:
# - Readability score
# - WCAG contrast compliance
# - Layout validity
```

**Expected metrics** (target):
- Readability: 85-90%
- Aesthetics: 80-85%
- Contrast compliance: 95%+

---

## 🎨 Week 3: Demo & Testing

### Day 12-14: Launch Demo

```bash
# Launch Gradio interface
python app/quote_app.py \
    --model_path ./checkpoints/quote_v1/checkpoint-best

# Open browser to http://127.0.0.1:7860
```

**Test the model**:
1. Upload a background image
2. Enter a quote (10-200 characters)
3. Click "Generate Quote Design"
4. Review the result

---

## 💡 Tips for Success

### Data Quality
- **More is better**: If you can collect more data, do it
- **Diverse backgrounds**: Mix nature, urban, abstract, minimalist
- **Quote variety**: Short and long, different sentiments

### Training
- **Watch the loss**: Should decrease steadily
- **Early stopping**: If validation loss plateaus, stop early
- **Checkpointing**: Save frequently, you can resume if interrupted

### Common Issues

**Out of memory?**
```yaml
# Edit training/training_config.yaml
per_device_train_batch_size: 4  # Reduce from 8
gradient_accumulation_steps: 8  # Increase from 4
```

**Training too slow?**
```yaml
# Use fewer data
# Or use smaller model
model:
  base_model: "meta-llama/Llama-3.2-1B"  # Instead of 3B
```

**Poor results?**
- Check data quality
- Increase training epochs
- Adjust learning rate
- Add more diverse training data

---

## 📊 Project Structure

```
quote_overlay/
├── dataset/
│   ├── scripts/
│   │   ├── download_datasets.py      ← Day 1
│   │   ├── prepare_crello.py         ← Day 2-3
│   │   └── create_dataset.py         ← Day 4-5
│   └── data/                          ← Downloaded data
├── training/
│   ├── training_config.yaml           ← Configure here
│   └── train_quote_model.py           ← Day 6-10
├── evaluation/                         ← Day 11
├── app/
│   └── quote_app.py                   ← Day 12-14
├── checkpoints/                        ← Model saves here
└── README.md
```

---

## 🎯 Success Checklist

### Week 1 ✓
- [ ] Downloaded Crello dataset
- [ ] Downloaded Quotes-500K
- [ ] Filtered 5,000+ quote designs
- [ ] Converted to LLaVA format
- [ ] Verified train/val/test splits

### Week 2 ✓
- [ ] Started training
- [ ] Training loss decreasing
- [ ] Best checkpoint saved
- [ ] Evaluation metrics look good

### Week 3 ✓
- [ ] Demo app works
- [ ] Can generate quote overlays
- [ ] Results look aesthetic
- [ ] Ready for production use

---

## 🚨 Troubleshooting

**Problem**: Crello download fails
- **Solution**: Check internet connection, use VPN if blocked, or download manually from HuggingFace

**Problem**: Not enough quote designs filtered
- **Solution**: Relax filtering criteria in `prepare_crello.py`:
  ```bash
  python dataset/scripts/prepare_crello.py \
      --min_quote_length 5 \
      --max_quote_length 300 \
      --min_text_coverage 0.05
  ```

**Problem**: Training crashes with CUDA out of memory
- **Solution**: Reduce batch size or use gradient checkpointing

**Problem**: Model generates invalid JSON
- **Solution**: Train longer, or add JSON validation in post-processing

---

## 📞 Next Steps

After completing this quick start:

1. **Collect more data**: Add PKU PosterLayout dataset
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Set up API server for production
4. **Iterate**: Gather user feedback and improve

---

## 📚 Resources

- [Full README](./README.md)
- [Training Config](./training/training_config.yaml)
- [elem2design Paper](https://arxiv.org/abs/2412.19712)
- [Crello Dataset](https://huggingface.co/datasets/cyberagent/crello)

---

**Estimated Total Time**: 2-3 weeks
**Estimated Total Cost**: $200-500 (if using cloud GPUs)

Good luck! 🚀
