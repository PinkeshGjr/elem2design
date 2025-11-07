"""
Download required datasets for quote overlay training.

Datasets:
1. Crello - Graphic design dataset from CyberAgent
2. Quotes-500K - Large collection of quotes
"""

import os
import argparse
from pathlib import Path
import datasets
import pandas as pd
import urllib.request
from tqdm import tqdm


def download_crello(output_dir: str = "./data/crello"):
    """
    Download Crello dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
    """
    print("=" * 80)
    print("Downloading Crello Dataset")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset from Hugging Face...")
    print("This may take 10-30 minutes depending on your connection.")

    # Download all splits
    crello = datasets.load_dataset("cyberagent/crello", revision="4.0.0")

    # Save to disk
    print(f"\nSaving to {output_dir}...")
    crello.save_to_disk(output_dir)

    # Print statistics
    print("\n✓ Crello dataset downloaded successfully!")
    print(f"  - Train: {len(crello['train'])} designs")
    print(f"  - Validation: {len(crello['validation'])} designs")
    print(f"  - Test: {len(crello['test'])} designs")
    print(f"  - Total: {len(crello['train']) + len(crello['validation']) + len(crello['test'])} designs")

    return crello


def download_quotes(output_dir: str = "./data/quotes"):
    """
    Download Quotes-500K dataset from GitHub.

    Args:
        output_dir: Directory to save the quotes
    """
    print("\n" + "=" * 80)
    print("Downloading Quotes-500K Dataset")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    quotes_file = output_path / "quotes.csv"

    if quotes_file.exists():
        print(f"\n✓ Quotes already downloaded at {quotes_file}")
    else:
        print("\nDownloading quotes from GitHub...")
        url = "https://raw.githubusercontent.com/ShivaliGoel/Quotes-500K/main/quotes.csv"

        try:
            urllib.request.urlretrieve(url, quotes_file)
            print(f"✓ Downloaded to {quotes_file}")
        except Exception as e:
            print(f"✗ Error downloading quotes: {e}")
            print("You can manually download from:")
            print("https://github.com/ShivaliGoel/Quotes-500K")
            return None

    # Load and show statistics
    try:
        df = pd.read_csv(quotes_file)
        print(f"\n✓ Quotes dataset loaded successfully!")
        print(f"  - Total quotes: {len(df)}")

        if 'tags' in df.columns:
            print(f"  - Categories: {df['tags'].nunique()}")
            print(f"\nTop 10 categories:")
            print(df['tags'].value_counts().head(10))
    except Exception as e:
        print(f"Warning: Could not load quotes CSV: {e}")

    return quotes_file


def download_background_images(output_dir: str = "./data/backgrounds", num_images: int = 1000):
    """
    Download sample background images from Unsplash.

    Note: This is a placeholder. For production, use Unsplash API with proper attribution.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to download
    """
    print("\n" + "=" * 80)
    print("Background Images")
    print("=" * 80)

    print("\n⚠️  Note: Background image download requires Unsplash API key.")
    print("For now, we'll use backgrounds from the Crello dataset.")
    print("\nTo add custom backgrounds:")
    print("1. Sign up at https://unsplash.com/developers")
    print("2. Get your API key")
    print("3. Use the Unsplash API to download images")
    print("4. Or manually add images to:", output_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def verify_downloads(data_dir: str = "./data"):
    """
    Verify all datasets are downloaded correctly.

    Args:
        data_dir: Root data directory
    """
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)

    data_path = Path(data_dir)

    checks = {
        "Crello dataset": data_path / "crello" / "train",
        "Quotes dataset": data_path / "quotes" / "quotes.csv",
        "Background images": data_path / "backgrounds",
    }

    all_good = True
    for name, path in checks.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_good = False

    if all_good:
        print("\n" + "=" * 80)
        print("✓ All datasets downloaded successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run prepare_crello.py to filter quote-style designs")
        print("2. Run create_dataset.py to convert to training format")
    else:
        print("\n✗ Some datasets are missing. Please check the errors above.")

    return all_good


def main():
    parser = argparse.ArgumentParser(description="Download datasets for quote overlay training")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Root directory to save datasets")
    parser.add_argument("--skip_crello", action="store_true",
                       help="Skip downloading Crello dataset")
    parser.add_argument("--skip_quotes", action="store_true",
                       help="Skip downloading quotes")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Quote Overlay Dataset Downloader")
    print("=" * 80)
    print(f"\nDownloading to: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Download datasets
    if not args.skip_crello:
        download_crello(os.path.join(args.output_dir, "crello"))

    if not args.skip_quotes:
        download_quotes(os.path.join(args.output_dir, "quotes"))

    download_background_images(os.path.join(args.output_dir, "backgrounds"))

    # Verify
    verify_downloads(args.output_dir)


if __name__ == "__main__":
    main()
