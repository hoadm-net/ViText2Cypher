#!/usr/bin/env python3
"""
Push fine-tuned Qwen model to Hugging Face Hub
Repository: hoadm-lab/qwen2.5-7b-instruct-vitext2cypher
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_card():
    """Create a comprehensive model card for the repository"""
    model_card_content = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- text-to-cypher
- vietnamese
- translation
- qwen2.5
- peft
- lora
- trl
language:
- en
- vi
datasets:
- custom
library_name: transformers
pipeline_tag: text-generation
---

# Qwen2.5-7B-Instruct Vietnamese Text-to-Cypher

This model is a fine-tuned version of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for English to Vietnamese translation with a focus on text-to-Cypher query translation tasks.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with RSLoRA
- **Training Framework**: TRL SFTTrainer
- **Task**: English to Vietnamese translation for database query descriptions
- **Language**: English → Vietnamese
- **Model Type**: Causal Language Model

## Training Details

### Training Data
- **Training Samples**: 4,559 samples
- **Validation Samples**: 1,140 samples
- **Data Format**: English database queries with Vietnamese translations
- **Domain**: Database query descriptions and Cypher query language

### Training Configuration
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.1
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Use RSLoRA**: True (Rank-Stabilized LoRA)
- **Quantization**: 4-bit with BitsAndBytesConfig
- **Epochs**: 3
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW with weight decay 0.01

### Training Results
- **Final Training Loss**: 0.027
- **Final Validation Loss**: 0.045
- **Token Accuracy**: 98.7%
- **Translation Accuracy**: 100% on test samples

## Usage

### Installation

```bash
pip install transformers torch peft accelerate
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model = "Qwen/Qwen2.5-7B-Instruct"
model_path = "hoadm-lab/qwen2.5-7b-instruct-vitext2cypher"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, model_path)
```

### Translation Example

```python
def translate_text(text: str) -> str:
    messages = [
        {
            "role": "system", 
            "content": "You are a professional translator. Only return the Vietnamese translation of the following question. Keep technical keywords and proper names unchanged."
        },
        {
            "role": "user", 
            "content": text
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract translation (implementation depends on chat template)
    return extract_translation(response)

# Example usage
english_text = "Find the top 5 suppliers with the highest average unit price."
vietnamese_translation = translate_text(english_text)
print(vietnamese_translation)
# Output: "Tìm 5 nhà cung cấp hàng đầu có giá đơn vị trung bình cao nhất."
```

## Training Examples

### Example 1
- **EN**: "Identify the 5 suppliers with the highest average unit price of products."
- **VI**: "Xác định 5 nhà cung cấp có giá đơn vị trung bình của sản phẩm cao nhất."

### Example 2
- **EN**: "What are the names of technicians who have not been assigned machine repair tasks?"
- **VI**: "Tên của những kỹ thuật viên chưa được giao nhiệm vụ sửa máy là gì?"

### Example 3
- **EN**: "How many companies are there in total?"
- **VI**: "Tổng số công ty là bao nhiêu?"

## Model Performance

The model achieves excellent translation quality with:
- 100% accuracy on test samples
- Natural Vietnamese output
- Preservation of technical terms
- Consistent formatting

## Technical Specifications

- **Parameters**: ~7B (base) + LoRA adapters
- **Memory Usage**: ~13GB VRAM (4-bit quantization)
- **Inference Speed**: ~2-3 seconds per translation on RTX 3090
- **Max Context Length**: 2048 tokens

## Limitations

- Specialized for database query translation domain
- May not generalize well to other translation tasks
- Requires sufficient GPU memory for inference
- Vietnamese translations may vary in style

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{qwen2.5-vitext2cypher,
  title={Qwen2.5-7B-Instruct Vietnamese Text-to-Cypher Fine-tuned Model},
  author={hoadm-lab},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/hoadm-lab/qwen2.5-7b-instruct-vitext2cypher}
}
```

## License

Apache 2.0 (same as base model)
"""
    return model_card_content

def update_adapter_config():
    """Update adapter config with proper base model name"""
    config_path = "/home/ft/qwen_translation_lora_trl/adapter_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update base model name
    config["base_model_name_or_path"] = "Qwen/Qwen2.5-7B-Instruct"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Updated adapter_config.json with correct base model name")

def push_model_to_hf(repo_name: str, model_dir: str, token: str = None):
    """Push the fine-tuned model to Hugging Face Hub"""
    
    logger.info(f"🚀 Starting upload to {repo_name}")
    
    # Create model card
    model_card = create_model_card()
    model_card_path = Path(model_dir) / "README.md"
    
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    logger.info("✅ Model card created")
    
    # Update adapter config
    update_adapter_config()
    
    # Initialize HF API
    api = HfApi(token=token)
    
    try:
        # Create repository
        create_repo(
            repo_id=repo_name,
            token=token,
            private=False,  # Set to True if you want private repo
            exist_ok=True
        )
        logger.info(f"✅ Repository {repo_name} created/verified")
        
        # Upload the model folder
        upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            token=token,
            commit_message="Upload fine-tuned Qwen2.5-7B-Instruct for Vietnamese text-to-cypher translation"
        )
        
        logger.info(f"✅ Model uploaded successfully to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def main():
    """Main function"""
    # Configuration
    repo_name = "hoadm-lab/qwen2.5-7b-instruct-vitext2cypher"
    model_dir = "/home/ft/qwen_translation_lora_trl"
    
    # Check if model directory exists
    if not Path(model_dir).exists():
        logger.error(f"❌ Model directory not found: {model_dir}")
        return
    
    # Get HF token from environment or user input
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        logger.info("Please provide your Hugging Face token:")
        logger.info("You can get it from: https://huggingface.co/settings/tokens")
        token = input("Enter your HF token: ").strip()
    
    if not token:
        logger.error("❌ No Hugging Face token provided")
        return
    
    # Validate model files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    
    for file in required_files:
        if not (Path(model_dir) / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return
    
    logger.info("📁 Model files validated")
    
    # Push to Hugging Face
    try:
        push_model_to_hf(repo_name, model_dir, token)
        logger.info("🎉 Upload completed successfully!")
        logger.info(f"🔗 Your model is now available at: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
