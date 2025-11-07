"""
Convert filtered Crello data to LLaVA training format.

Input: Filtered Crello JSON files
Output: LLaVA conversation format JSON for training
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def extract_background_element(sample: Dict) -> Dict:
    """
    Extract the background image element.

    Args:
        sample: Filtered Crello sample

    Returns:
        Background element info
    """
    # Find first image element (type 0)
    for i, elem_type in enumerate(sample['type']):
        if elem_type == 0:  # Image
            return {
                'index': i,
                'left': sample['left'][i],
                'top': sample['top'][i],
                'width': sample['width'][i],
                'height': sample['height'][i],
                'image': sample['image'][i]
            }

    return None


def extract_text_elements(sample: Dict) -> List[Dict]:
    """
    Extract all text elements.

    Args:
        sample: Filtered Crello sample

    Returns:
        List of text element info
    """
    text_elements = []

    for i, elem_type in enumerate(sample['type']):
        if elem_type == 1 and sample['text'][i].strip():  # Text, non-empty
            text_elements.append({
                'index': i,
                'text': sample['text'][i],
                'left': sample['left'][i],
                'top': sample['top'][i],
                'width': sample['width'][i],
                'height': sample['height'][i],
                'color': sample['color'][i] if i < len(sample['color']) else '#000000',
                'angle': sample['angle'][i] if i < len(sample['angle']) else 0
            })

    return text_elements


def normalize_bbox(bbox: Dict, canvas_width: int, canvas_height: int) -> Dict:
    """
    Normalize bounding box to 0-1 range.

    Args:
        bbox: Bounding box with left, top, width, height
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels

    Returns:
        Normalized bbox
    """
    return {
        'left': bbox['left'] / canvas_width,
        'top': bbox['top'] / canvas_height,
        'width': bbox['width'] / canvas_width,
        'height': bbox['height'] / canvas_height
    }


def rgb_to_hex(rgb_tuple) -> str:
    """
    Convert RGB tuple to hex color.

    Args:
        rgb_tuple: (r, g, b) or (r, g, b, a)

    Returns:
        Hex color string
    """
    if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) >= 3:
        r, g, b = rgb_tuple[:3]
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return "#000000"  # Default black


def create_design_params(sample: Dict) -> Dict:
    """
    Create design parameters in the output format.

    Args:
        sample: Filtered Crello sample

    Returns:
        Design parameters JSON
    """
    text_elements = extract_text_elements(sample)

    if not text_elements:
        return None

    # Use the largest text element as main text
    main_text = max(text_elements, key=lambda x: x['width'] * x['height'])

    # Normalize bbox
    normalized_bbox = normalize_bbox(
        {
            'left': main_text['left'],
            'top': main_text['top'],
            'width': main_text['width'],
            'height': main_text['height']
        },
        sample['canvas_width'],
        sample['canvas_height']
    )

    # Create design parameters
    design_params = {
        "text_box": normalized_bbox,
        "text_style": {
            "font_family": "Arial",  # Crello doesn't always provide font info
            "font_size": int(main_text['height'] * 0.8),  # Approximate
            "font_weight": "bold",
            "text_color": rgb_to_hex(main_text['color']),
            "text_align": "center",
            "line_height": 1.4
        },
        "background_overlay": {
            "type": "solid_box",
            "color": "#000000",
            "opacity": 0.5,
            "padding": 20,
            "border_radius": 8
        },
        "effects": {
            "text_shadow": {
                "enabled": True,
                "color": "#000000",
                "blur": 4,
                "offset_x": 2,
                "offset_y": 2
            },
            "text_stroke": {
                "enabled": False,
                "color": "#000000",
                "width": 2
            }
        }
    }

    return design_params


def convert_to_llava_format(sample: Dict, image_folder: str) -> Dict:
    """
    Convert Crello sample to LLaVA conversation format.

    Args:
        sample: Filtered Crello sample
        image_folder: Path to image folder

    Returns:
        LLaVA format sample
    """
    background = extract_background_element(sample)

    if not background:
        return None

    # Get design parameters
    design_params = create_design_params(sample)

    if not design_params:
        return None

    # Create conversation
    conversation = {
        "id": sample['id'],
        "image": [background['image']['path']],  # Path to background image
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given this background image: <image>\n"
                    f"Place the following quote aesthetically: '{sample['quote']}'\n"
                    f"Consider the image's color palette, composition, and mood. "
                    f"Predict the optimal design parameters."
                )
            },
            {
                "from": "gpt",
                "value": json.dumps(design_params, ensure_ascii=False)
            }
        ],
        "canvas_width": sample['canvas_width'],
        "canvas_height": sample['canvas_height'],
        "quote": sample['quote']
    }

    return conversation


def create_llava_dataset(
    filtered_dir: str,
    output_dir: str,
    image_folder: str = "./data/crello_images"
):
    """
    Convert filtered Crello data to LLaVA training format.

    Args:
        filtered_dir: Directory with filtered JSON files
        output_dir: Output directory for LLaVA format data
        image_folder: Path to Crello images
    """
    print("=" * 80)
    print("Converting to LLaVA Training Format")
    print("=" * 80)

    filtered_path = Path(filtered_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    statistics = {
        'total': 0,
        'converted': 0,
        'skipped': 0
    }

    # Process each split
    for split in ['train', 'validation', 'test']:
        input_file = filtered_path / f"{split}_filtered.json"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping {split}")
            continue

        print(f"\nProcessing {split} split...")

        # Load filtered data
        with open(input_file, 'r') as f:
            filtered_data = json.load(f)

        statistics['total'] += len(filtered_data)

        # Convert to LLaVA format
        llava_data = []

        for sample in tqdm(filtered_data):
            conversation = convert_to_llava_format(sample, image_folder)

            if conversation:
                llava_data.append(conversation)
                statistics['converted'] += 1
            else:
                statistics['skipped'] += 1

        # Save LLaVA format data
        output_file = output_path / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(llava_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(llava_data)} samples to {output_file}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Conversion Statistics")
    print("=" * 80)
    print(f"Total samples: {statistics['total']}")
    print(f"Successfully converted: {statistics['converted']}")
    print(f"Skipped: {statistics['skipped']}")
    print(f"Success rate: {statistics['converted']/statistics['total']*100:.1f}%")

    # Save statistics
    stats_file = output_path / "conversion_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"\n✓ Statistics saved to {stats_file}")

    # Create sample preview
    create_sample_preview(output_path)

    print("\n" + "=" * 80)
    print("✓ Conversion complete!")
    print("=" * 80)
    print("\nYour training data is ready!")
    print(f"Location: {output_dir}")
    print("\nNext step: Train the model with train_quote_model.py")


def create_sample_preview(output_dir: Path):
    """
    Create a preview of sample conversations.

    Args:
        output_dir: Output directory
    """
    train_file = output_dir / "train.json"

    if not train_file.exists():
        return

    # Load first 3 samples
    with open(train_file, 'r') as f:
        data = json.load(f)

    preview_file = output_dir / "sample_preview.txt"

    with open(preview_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Sample Training Conversations (First 3)\n")
        f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(data[:3]):
            f.write(f"Sample {i+1}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"ID: {sample['id']}\n")
            f.write(f"Quote: {sample['quote']}\n")
            f.write(f"Canvas: {sample['canvas_width']}x{sample['canvas_height']}\n")
            f.write(f"Image: {sample['image'][0]}\n\n")

            f.write("Conversation:\n")
            for turn in sample['conversations']:
                f.write(f"{turn['from'].upper()}:\n")
                f.write(f"{turn['value']}\n\n")

            f.write("\n")

    print(f"✓ Sample preview saved to {preview_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert filtered Crello to LLaVA format")
    parser.add_argument("--filtered_dir", type=str, default="./data/filtered",
                       help="Directory with filtered JSON files")
    parser.add_argument("--output_dir", type=str, default="./data/llava_format",
                       help="Output directory for LLaVA format data")
    parser.add_argument("--image_folder", type=str, default="./data/crello_images",
                       help="Path to Crello images folder")
    args = parser.parse_args()

    create_llava_dataset(
        filtered_dir=args.filtered_dir,
        output_dir=args.output_dir,
        image_folder=args.image_folder
    )


if __name__ == "__main__":
    main()
