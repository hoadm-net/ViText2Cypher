#!/usr/bin/env python3
"""
Script to randomly sample 2000 data points from train.json and save to data.json
"""

import argparse
from mint import DataSampler

def main():
    parser = argparse.ArgumentParser(description='Sample random data points from dataset')
    parser.add_argument('--input', default='dataset/train.json', help='Input file (default: dataset/train.json)')
    parser.add_argument('--output', default='dataset/data.json', help='Output file (default: dataset/data.json)')
    parser.add_argument('--size', type=int, default=2000, help='Sample size (default: 2000)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Initialize sampler with optional seed
    sampler = DataSampler(random_seed=args.seed)

    # Sample data from file
    sampler.sample_from_file(args.input, args.output, args.size)

if __name__ == "__main__":
    main()
