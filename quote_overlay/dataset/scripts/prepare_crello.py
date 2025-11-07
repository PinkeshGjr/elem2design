"""
Filter Crello dataset for quote-style designs.

Criteria for quote-style designs:
1. Has 1-2 background image elements
2. Has 1-3 text elements (quote + optional author)
3. Simple composition (not too many elements)
4. Text is the primary focus
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List
import datasets
from tqdm import tqdm
import pandas as pd


def is_quote_style_design(sample: Dict) -> bool:
    """
    Determine if a design is quote-style.

    Args:
        sample: Crello dataset sample

    Returns:
        True if design appears to be a quote overlay
    """
    length = sample['length']

    # Get element types
    types = sample['type']
    texts = sample['text']

    # Count elements by type
    num_images = sum(1 for t in types if t == 0)  # Type 0 = image
    num_texts = sum(1 for t, text in zip(types, texts) if t == 1 and text.strip())  # Type 1 = text, non-empty
    num_shapes = sum(1 for t in types if t == 2)  # Type 2 = shape

    # Quote-style criteria
    criteria = {
        "total_elements": 2 <= length <= 8,  # Not too complex
        "has_background": num_images >= 1,   # At least one background
        "has_text": 1 <= num_texts <= 4,     # 1-4 text elements
        "not_too_busy": num_shapes <= 3,     # Simple overlay
        "text_focused": num_texts >= num_images,  # Text is main content
    }

    # Must pass all criteria
    return all(criteria.values())


def extract_quote_text(sample: Dict) -> str:
    """
    Extract and combine text elements into a quote.

    Args:
        sample: Crello dataset sample

    Returns:
        Combined quote text
    """
    texts = []
    for elem_type, text in zip(sample['type'], sample['text']):
        if elem_type == 1 and text.strip():  # Text element, non-empty
            texts.append(text.strip())

    return " ".join(texts)


def calculate_text_coverage(sample: Dict) -> float:
    """
    Calculate what percentage of canvas is covered by text.

    Args:
        sample: Crello dataset sample

    Returns:
        Coverage ratio (0-1)
    """
    canvas_width = sample['canvas_width']
    canvas_height = sample['canvas_height']
    canvas_area = canvas_width * canvas_height

    text_area = 0
    for elem_type, width, height in zip(sample['type'], sample['width'], sample['height']):
        if elem_type == 1:  # Text element
            text_area += width * height

    return text_area / canvas_area if canvas_area > 0 else 0


def filter_crello_dataset(
    crello_path: str,
    output_dir: str,
    min_quote_length: int = 10,
    max_quote_length: int = 200,
    min_text_coverage: float = 0.1,
    max_text_coverage: float = 0.7
):
    """
    Filter Crello dataset for quote-style designs.

    Args:
        crello_path: Path to downloaded Crello dataset
        output_dir: Directory to save filtered data
        min_quote_length: Minimum character count for quote
        max_quote_length: Maximum character count for quote
        min_text_coverage: Minimum text coverage of canvas
        max_text_coverage: Maximum text coverage of canvas
    """
    print("=" * 80)
    print("Filtering Crello Dataset for Quote-Style Designs")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading Crello dataset from {crello_path}...")
    crello = datasets.load_from_disk(crello_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filtered_data = {
        'train': [],
        'validation': [],
        'test': []
    }

    statistics = {
        'total': 0,
        'quote_style': 0,
        'quote_length_ok': 0,
        'text_coverage_ok': 0,
        'final_filtered': 0
    }

    # Process each split
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")

        for sample in tqdm(crello[split]):
            statistics['total'] += 1

            # Check if quote-style
            if not is_quote_style_design(sample):
                continue
            statistics['quote_style'] += 1

            # Extract quote text
            quote = extract_quote_text(sample)

            # Check quote length
            if not (min_quote_length <= len(quote) <= max_quote_length):
                continue
            statistics['quote_length_ok'] += 1

            # Check text coverage
            coverage = calculate_text_coverage(sample)
            if not (min_text_coverage <= coverage <= max_text_coverage):
                continue
            statistics['text_coverage_ok'] += 1

            # Passed all filters
            statistics['final_filtered'] += 1

            # Save filtered sample
            filtered_sample = {
                'id': sample['id'],
                'quote': quote,
                'canvas_width': sample['canvas_width'],
                'canvas_height': sample['canvas_height'],
                'length': sample['length'],
                'type': sample['type'],
                'left': sample['left'],
                'top': sample['top'],
                'width': sample['width'],
                'height': sample['height'],
                'image': sample['image'],
                'text': sample['text'],
                'color': sample['color'],
                'angle': sample['angle'],
                'text_coverage': coverage
            }

            filtered_data[split].append(filtered_sample)

    # Print statistics
    print("\n" + "=" * 80)
    print("Filtering Statistics")
    print("=" * 80)
    print(f"Total designs processed: {statistics['total']}")
    print(f"Quote-style designs: {statistics['quote_style']} ({statistics['quote_style']/statistics['total']*100:.1f}%)")
    print(f"With valid quote length: {statistics['quote_length_ok']} ({statistics['quote_length_ok']/statistics['total']*100:.1f}%)")
    print(f"With valid text coverage: {statistics['text_coverage_ok']} ({statistics['text_coverage_ok']/statistics['total']*100:.1f}%)")
    print(f"Final filtered: {statistics['final_filtered']} ({statistics['final_filtered']/statistics['total']*100:.1f}%)")

    print(f"\nSplit breakdown:")
    print(f"  Train: {len(filtered_data['train'])} samples")
    print(f"  Validation: {len(filtered_data['validation'])} samples")
    print(f"  Test: {len(filtered_data['test'])} samples")

    # Save filtered data
    for split in ['train', 'validation', 'test']:
        output_file = output_path / f"{split}_filtered.json"
        with open(output_file, 'w') as f:
            json.dump(filtered_data[split], f, indent=2)
        print(f"\n✓ Saved {split} to {output_file}")

    # Save statistics
    stats_file = output_path / "filtering_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved statistics to {stats_file}")

    # Create analysis report
    create_analysis_report(filtered_data, output_path)

    print("\n" + "=" * 80)
    print("✓ Filtering complete!")
    print("=" * 80)
    print("\nNext step: Run create_dataset.py to convert to training format")

    return filtered_data


def create_analysis_report(filtered_data: Dict, output_dir: Path):
    """
    Create detailed analysis report of filtered data.

    Args:
        filtered_data: Filtered dataset
        output_dir: Output directory
    """
    print("\nCreating analysis report...")

    # Combine all splits for analysis
    all_samples = []
    for split in ['train', 'validation', 'test']:
        all_samples.extend(filtered_data[split])

    # Analyze quote lengths
    quote_lengths = [len(s['quote']) for s in all_samples]

    # Analyze text coverage
    text_coverages = [s['text_coverage'] for s in all_samples]

    # Analyze canvas sizes
    canvas_sizes = [(s['canvas_width'], s['canvas_height']) for s in all_samples]

    # Analyze number of elements
    num_elements = [s['length'] for s in all_samples]

    # Create report
    report = {
        "total_samples": len(all_samples),
        "quote_length": {
            "min": min(quote_lengths),
            "max": max(quote_lengths),
            "mean": sum(quote_lengths) / len(quote_lengths),
            "median": sorted(quote_lengths)[len(quote_lengths) // 2]
        },
        "text_coverage": {
            "min": min(text_coverages),
            "max": max(text_coverages),
            "mean": sum(text_coverages) / len(text_coverages),
            "median": sorted(text_coverages)[len(text_coverages) // 2]
        },
        "num_elements": {
            "min": min(num_elements),
            "max": max(num_elements),
            "mean": sum(num_elements) / len(num_elements),
            "median": sorted(num_elements)[len(num_elements) // 2]
        },
        "canvas_sizes": {
            "unique_sizes": len(set(canvas_sizes)),
            "most_common": max(set(canvas_sizes), key=canvas_sizes.count)
        }
    }

    # Save report
    report_file = output_dir / "analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Analysis report saved to {report_file}")

    # Print summary
    print("\nDataset Analysis:")
    print(f"  Quote length: {report['quote_length']['min']}-{report['quote_length']['max']} chars (avg: {report['quote_length']['mean']:.1f})")
    print(f"  Text coverage: {report['text_coverage']['min']:.2%}-{report['text_coverage']['max']:.2%} (avg: {report['text_coverage']['mean']:.2%})")
    print(f"  Elements per design: {report['num_elements']['min']}-{report['num_elements']['max']} (avg: {report['num_elements']['mean']:.1f})")
    print(f"  Most common canvas size: {report['canvas_sizes']['most_common']}")


def main():
    parser = argparse.ArgumentParser(description="Filter Crello dataset for quote-style designs")
    parser.add_argument("--crello_path", type=str, default="./data/crello",
                       help="Path to Crello dataset")
    parser.add_argument("--output_dir", type=str, default="./data/filtered",
                       help="Output directory for filtered data")
    parser.add_argument("--min_quote_length", type=int, default=10,
                       help="Minimum quote length in characters")
    parser.add_argument("--max_quote_length", type=int, default=200,
                       help="Maximum quote length in characters")
    parser.add_argument("--min_text_coverage", type=float, default=0.1,
                       help="Minimum text coverage (0-1)")
    parser.add_argument("--max_text_coverage", type=float, default=0.7,
                       help="Maximum text coverage (0-1)")
    args = parser.parse_args()

    filter_crello_dataset(
        crello_path=args.crello_path,
        output_dir=args.output_dir,
        min_quote_length=args.min_quote_length,
        max_quote_length=args.max_quote_length,
        min_text_coverage=args.min_text_coverage,
        max_text_coverage=args.max_text_coverage
    )


if __name__ == "__main__":
    main()
