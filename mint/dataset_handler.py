"""
Dataset handling utilities for loading and processing data from Hugging Face
"""

import json
import os
from datasets import load_dataset
from typing import List, Dict, Any

class DatasetHandler:
    """Handle dataset operations including loading from Hugging Face and saving locally."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)

    def load_from_huggingface(self, dataset_name: str = "neo4j/text2cypher-2025v1") -> Dict[str, List[Dict]]:
        """
        Load dataset from Hugging Face and return train/test splits.

        Args:
            dataset_name: Name of the dataset on Hugging Face

        Returns:
            Dictionary containing train and test data
        """
        print(f"Loading dataset '{dataset_name}' from Hugging Face...")

        dataset = load_dataset(dataset_name)

        # Extract train data
        train_data = []
        for item in dataset['train']:
            train_data.append({
                'question': item['question'],
                'schema': item['schema'],
                'cypher': item['cypher']
            })

        # Extract test data
        test_data = []
        for item in dataset['test']:
            test_data.append({
                'question': item['question'],
                'schema': item['schema'],
                'cypher': item['cypher']
            })

        print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")

        return {
            'train': train_data,
            'test': test_data
        }

    def save_to_json(self, data: Dict[str, List[Dict]], train_filename: str = "train.json",
                     test_filename: str = "test.json") -> None:
        """
        Save train and test data to JSON files.

        Args:
            data: Dictionary containing train and test data
            train_filename: Filename for training data
            test_filename: Filename for test data
        """
        # Save train data
        train_path = os.path.join(self.dataset_dir, train_filename)
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(data['train'], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data['train'])} training samples to {train_path}")

        # Save test data
        test_path = os.path.join(self.dataset_dir, test_filename)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(data['test'], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data['test'])} test samples to {test_path}")

    def load_and_save_dataset(self, dataset_name: str = "neo4j/text2cypher-2025v1") -> None:
        """
        Complete workflow: load from Hugging Face and save locally.

        Args:
            dataset_name: Name of the dataset on Hugging Face
        """
        data = self.load_from_huggingface(dataset_name)
        self.save_to_json(data)
        print("Dataset loading completed!")

    def load_json_file(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            filename: Name of the JSON file to load

        Returns:
            List of data samples
        """
        filepath = os.path.join(self.dataset_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
