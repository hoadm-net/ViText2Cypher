#!/usr/bin/env python3
"""
Script to load the text2cypher dataset from Hugging Face and save it locally.
"""

from mint import DatasetHandler

def main():
    """Main function to load and save dataset."""
    handler = DatasetHandler()
    handler.load_and_save_dataset()

if __name__ == "__main__":
    main()
