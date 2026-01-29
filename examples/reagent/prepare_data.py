"""
Data preparation script for training multi-tool agent.

This script loads and preprocesses multiple datasets for training:
- Math reasoning data (DeepScaleR)
- Audio QA data (HeySQuAD)
- Web search data (Hiersearch)
- Image QA data (OCR_VQA)
- Test data (AIME 2024)

Usage:
    python prepare_data.py
"""

import os
from datasets import load_dataset, concatenate_datasets
from rllm.data.dataset import DatasetRegistry

# Configuration: Update these paths to your data locations
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/your/data")  # TODO: Set your data root directory

# Training data paths
MATH_DATA_PATH = os.path.join(DATA_ROOT, "DeepScaleR_18_6k_5k.json")          # Math reasoning dataset
AUDIO_DATA_PATH = os.path.join(DATA_ROOT, "HeySQuAD_4k_2k_converted.json")    # Audio QA dataset
SEARCH_DATA_PATH = os.path.join(DATA_ROOT, "Hiersearch_13k.json")             # Web search dataset
IMAGE_DATA_PATH = os.path.join(DATA_ROOT, "OCR_VQA_8k_5k_converted.json")     # Image QA dataset

# Test data path
TEST_DATA_PATH = os.path.join(DATA_ROOT, "aime24.jsonl")  # AIME 2024 test set


def prepare_math_data():
    """
    Prepare training and test datasets for multi-tool agent training.
    
    Returns:
        tuple: (train_dataset, test_dataset) - Preprocessed datasets ready for training
    """
    # Load training datasets
    print("Loading training datasets...")
    train1 = load_dataset("json", data_files=MATH_DATA_PATH, split="train")    # Math reasoning
    train2 = load_dataset("json", data_files=AUDIO_DATA_PATH, split="train")   # Audio QA
    train3 = load_dataset("json", data_files=SEARCH_DATA_PATH, split="train")  # Web search
    train4 = load_dataset("json", data_files=IMAGE_DATA_PATH, split="train")   # Image QA
    
    # Load test dataset
    print("Loading test dataset...")
    test1 = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

    # Helper function to drop unnecessary columns
    def drop_multimodal_info(ds):
        """Remove multimodal_info and meta_info columns if they exist."""
        cols = [c for c in ["multimodal_info", "meta_info"] if c in ds.column_names]
        return ds.remove_columns(cols) if cols else ds

    # Clean up training datasets
    print("Cleaning datasets...")
    train1 = drop_multimodal_info(train1)
    train2 = drop_multimodal_info(train2)
    train3 = drop_multimodal_info(train3)
    train4 = drop_multimodal_info(train4)
    test1 = drop_multimodal_info(test1)
    
    # Preprocessing functions
    def preprocess_generic(example, idx, data_source):
        """
        Generic preprocessing function for all data sources.
        
        Args:
            example: A dataset example with question, ground_truth fields
            idx: Index of the example
            data_source: String indicating the data source type (e.g., "math", "audio", "search", "image")
        
        Returns:
            dict: Preprocessed task dictionary
        """
        meta_info = example.get("meta_info", {})
        meta_source = meta_info.get("source", "")
        category = example.get("category", "")
        
        task_dict = {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "data_source": data_source,
            "meta_info_source": meta_source or "",
            "category": category or "",
        }
        
        return task_dict
    
    def preprocess_aime(example, idx):
        """
        Preprocess AIME dataset (test set).
        
        Args:
            example: A dataset example with problem and answer fields
            idx: Index of the example
        
        Returns:
            dict: Preprocessed task dictionary
        """
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    # Apply preprocessing to each dataset
    print("Preprocessing datasets...")
    train1 = train1.map(lambda x, idx: preprocess_generic(x, idx, "math"), with_indices=True)     # Math reasoning
    train2 = train2.map(lambda x, idx: preprocess_generic(x, idx, "audio"), with_indices=True)    # Audio QA
    train3 = train3.map(lambda x, idx: preprocess_generic(x, idx, "search"), with_indices=True)   # Web search
    train4 = train4.map(lambda x, idx: preprocess_generic(x, idx, "image"), with_indices=True)    # Image QA
    
    # Concatenate all training datasets
    print("Concatenating training datasets...")
    train_dataset = concatenate_datasets([train1, train2, train3, train4])
    print(f"Total training examples: {len(train_dataset)}")
    
    # Preprocess test dataset
    print("Preprocessing test dataset...")
    test_dataset = test1.map(preprocess_aime, with_indices=True)
    print(f"Total test examples: {len(test_dataset)}")
    
    # Register datasets with rLLM registry
    print("Registering datasets...")
    train_dataset = DatasetRegistry.register_dataset("reagent", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    
    print("Dataset preparation completed successfully!")
    return train_dataset, test_dataset


if __name__ == "__main__":
    """Main entry point for data preparation."""
    print("=" * 80)
    print("Starting data preparation for multi-tool agent training")
    print("=" * 80)
    
    train_dataset, test_dataset = prepare_math_data()
    
    print("\n" + "=" * 80)
    print("Dataset preparation summary:")
    print(f"  Training dataset: {len(train_dataset)} examples")
    print(f"  Test dataset: {len(test_dataset)} examples")
    print("=" * 80)
