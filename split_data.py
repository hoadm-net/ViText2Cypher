#!/usr/bin/env python3
"""
Split data.json into 4 parts with 500 samples each
"""

import json
import os

def clean_cypher(sample):
    """Replace \\n with \n in Cypher queries"""
    if 'cypher' in sample:
        sample['cypher'] = sample['cypher'].replace('\\n', '\n')
    return sample

def split_data():
    """Split data.json into 4 parts: data_p1.json, data_p2.json, data_p3.json, data_p4.json"""

    # Load the original data
    input_path = "dataset/data.json"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    print("Loading data.json...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total samples: {total_samples}")

    # Clean the data - replace \\n with \n in Cypher queries
    print("Cleaning Cypher queries...")
    data = [clean_cypher(sample) for sample in data]

    # Define the splits
    splits = [
        (0, 500, "data_p1.json"),      # 0-499
        (500, 1000, "data_p2.json"),   # 500-999
        (1000, 1500, "data_p3.json"),  # 1000-1499
        (1500, None, "data_p4.json")   # 1500 to end
    ]

    # Create each split
    for start, end, filename in splits:
        if end is None:
            end = total_samples

        split_data = data[start:end]
        output_path = f"dataset/{filename}"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        print(f"Created {filename} with {len(split_data)} samples (indices {start}-{end-1})")

if __name__ == "__main__":
    split_data()
