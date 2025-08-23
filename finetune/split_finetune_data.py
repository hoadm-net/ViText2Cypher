#!/usr/bin/env python3
import json
import random
from pathlib import Path

def create_message_format(question, schema, translation):
    """
    Create message format as required
    """
    return {
        "messages": [
            {
                "role": "system", 
                "content": "You are a professional translator. Only return the Vietnamese translation of the content in <QUESTION>. Completely ignore <SCHEMA>, the content in <SCHEMA> is only for supporting what needs to be translated."
            },
            {
                "role": "user",
                "content": f"<QUESTION>{question}</QUESTION>\n<SCHEMA>{schema}</SCHEMA>"
            },
            {
                "role": "assistant",
                "content": translation
            }
        ]
    }

def split_data():
    # Read original data
    input_file = Path("../data/augmented_data.json")
    
    if not input_file.exists():
        print(f"File {input_file} does not exist!")
        return
    
    print("Reading data from augmented_data.json...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Randomly shuffle data
    random.seed(42)  # For consistent results
    random.shuffle(data)
    
    # Split 80:20 for train:dev
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    
    # Create message format and write JSONL files
    print("Creating ../data/ft_train.jsonl...")
    with open("../data/ft_train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            message_data = create_message_format(
                item['question'], 
                item['schema'], 
                item['translation']
            )
            f.write(json.dumps(message_data, ensure_ascii=False) + '\n')
    
    print("Creating ../data/ft_dev.jsonl...")
    with open("../data/ft_dev.jsonl", 'w', encoding='utf-8') as f:
        for item in dev_data:
            message_data = create_message_format(
                item['question'], 
                item['schema'], 
                item['translation']
            )
            f.write(json.dumps(message_data, ensure_ascii=False) + '\n')
    
    print("Completed!")
    print(f"- ../data/ft_train.jsonl: {len(train_data)} samples")
    print(f"- ../data/ft_dev.jsonl: {len(dev_data)} samples")
    
    # Show example
    print("\nExample sample in ft_train.jsonl:")
    example = create_message_format(
        train_data[0]['question'], 
        train_data[0]['schema'], 
        train_data[0]['translation']
    )
    print(json.dumps(example, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    split_data()
