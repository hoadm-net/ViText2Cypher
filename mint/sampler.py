"""
Data sampling utilities for random sampling and data processing
"""

import json
import random
from typing import List, Dict, Any, Optional

class DataSampler:
    """Handle data sampling operations."""

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize DataSampler.

        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
            self.random_seed = random_seed
        else:
            self.random_seed = None

    def sample_data(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        Sample random data points from input data.

        Args:
            data: Input data list
            sample_size: Number of samples to extract

        Returns:
            List of sampled data points
        """
        if len(data) < sample_size:
            print(f"Warning: Dataset has only {len(data)} samples, less than requested {sample_size}")
            return data
        else:
            return random.sample(data, sample_size)

    def sample_from_file(self, input_file: str, output_file: str, sample_size: int = 2000) -> None:
        """
        Sample random data points from input file and save to output file.

        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            sample_size: Number of samples to extract
        """
        print(f"Loading data from {input_file}...")

        # Load the data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples")

        # Sample random data points
        sampled_data = self.sample_data(data, sample_size)

        # Save sampled data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(sampled_data)} randomly sampled data points to {output_file}")

        if self.random_seed is not None:
            print(f"Used random seed: {self.random_seed}")

    def get_data_slice(self, data: List[Dict[str, Any]], start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a slice of data from start to end position.

        Args:
            data: Input data list
            start: Start position
            end: End position (if None, goes to end of data)

        Returns:
            Sliced data
        """
        if end is None:
            end = len(data)

        return data[start:end]
