#!/usr/bin/env python3
"""
Script to translate English questions to Vietnamese for Cypher queries
"""

import argparse
from mint import CypherTranslator

def main():
    parser = argparse.ArgumentParser(description='Translate questions from English to Vietnamese for Cypher queries')
    parser.add_argument('--start', type=int, default=0, help='Start position (default: 0)')
    parser.add_argument('--end', type=int, help='End position (default: end of dataset)')

    args = parser.parse_args()

    # Auto-generate output file name
    if args.end is None:
        output_file = f'dataset/translated_data_{args.start}_end.json'
    else:
        output_file = f'dataset/translated_data_{args.start}_{args.end}.json'

    # Initialize translator
    translator = CypherTranslator()

    # Translate and save
    translator.translate_and_save(
        input_file='dataset/data.json',
        output_file=output_file,
        start=args.start,
        end=args.end
    )

if __name__ == "__main__":
    main()
