from datasets import load_dataset
from typing import Optional
import os
import json

def download_dataset(
    dataset_name: str,
    output_path: str,
    subset: Optional[str] = None,
    data_dir: Optional[str] = None
) -> None:
    """
    Download a dataset from HuggingFace and save it to JSON.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        output_path: Path where to save the JSON file
        subset: Optional subset name of the dataset
        data_dir: Optional data directory for the dataset
    """
    # Load dataset with optional parameters
    dataset = load_dataset(
        dataset_name,
        subset,
        data_dir=data_dir if data_dir else None
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save each split to a separate JSON file
    if isinstance(dataset, dict):
        for split_name, split_dataset in dataset.items():
            split_path = output_path.replace('.json', f'_{split_name}.json')
            split_dataset.to_json(split_path)
    else:
        # If it's a single dataset, save directly
        dataset.to_json(output_path)

def main():
    # Example usage
    # Download full dataset
    download_dataset(
        "HuggingFaceH4/helpful_instructions",
        "helpful_instructions.json"
    )
    
    # Download specific subset
    download_dataset(
        "HuggingFaceH4/helpful_instructions",
        "helpful_instructions_subset.json",
        data_dir="data/helpful-anthropic-raw"
    )

if __name__ == "__main__":
    main()