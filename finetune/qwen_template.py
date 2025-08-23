#!/usr/bin/env python3
"""
Template Fine-tuning Qwen2.5-7B for Translation Task
Using TRL SFTTrainer with LoRA
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTranslationFineTuner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.output_dir = "./qwen_translation_model"
        
    def load_dataset(self, train_file: str, dev_file: str) -> tuple:
        """Load training data"""
        def load_jsonl(file_path: str):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        train_data = load_jsonl(train_file)
        dev_data = load_jsonl(dev_file)
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(dev_data)} validation samples")
        
        return Dataset.from_list(train_data), Dataset.from_list(dev_data)
    
    def create_configs(self):
        """Create LoRA and SFT configurations"""
        # LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
        )
        
        # SFT Configuration
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            weight_decay=0.01,
            warmup_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=400,
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_steps=10,
            max_length=2048,
            eos_token="<|im_end|>",
            assistant_only_loss=False,
            packing=False,
            padding_free=False,
            model_init_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": {"": 0},
                "trust_remote_code": True,
                "attn_implementation": "eager",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            }
        )
        
        return lora_config, sft_config
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Fine-tune model"""
        logger.info("Starting fine-tuning...")
        
        lora_config, sft_config = self.create_configs()
        
        trainer = SFTTrainer(
            model=self.model_name,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
        )
        
        if hasattr(trainer.model, 'print_trainable_parameters'):
            trainer.model.print_trainable_parameters()
        
        trainer.train()
        trainer.save_model()
        
        # Save tokenizer
        tokenizer = trainer.processing_class if hasattr(trainer, 'processing_class') else trainer.tokenizer
        if tokenizer:
            tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        return trainer

class QwenTranslator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned model"""
        logger.info("Loading model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        
        # Load base model with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # Load LoRA weights
        try:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            logger.info("LoRA weights loaded successfully")
        except:
            logger.warning("No LoRA weights found, using base model")
    
    def translate_text(self, text: str) -> str:
        """Translate English text to Vietnamese"""
        # Create chat template
        messages = [
            {"role": "system", "content": "You are a professional translator. Only return the Vietnamese translation of the following question. Keep technical keywords and proper names unchanged."},
            {"role": "user", "content": text}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            # Convert to float32 for stable generation
            model_float = self.model.float()
            outputs = model_float.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            # Convert back
            self.model = self.model.bfloat16()
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation - Fixed extraction logic
        if "<|im_start|>assistant\n" in response:
            translation = response.split("<|im_start|>assistant\n")[-1].strip()
            if translation.endswith("<|im_end|>"):
                translation = translation[:-10].strip()
        elif "assistant\n" in response:
            # Handle case where special tokens are removed
            translation = response.split("assistant\n")[-1].strip()
        else:
            # Fallback: try to find the response after the last user input
            if "user\n" in response:
                parts = response.split("user\n")
                if len(parts) > 1:
                    # Get everything after the last user input
                    last_part = parts[-1]
                    # Remove the original question if it's still there
                    if text in last_part:
                        translation = last_part.replace(text, "").strip()
                    else:
                        translation = last_part.strip()
                else:
                    translation = response.strip()
            else:
                translation = response.strip()
        
        return translation

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "translate"], required=True)
    parser.add_argument("--train_file", default="ft_train.jsonl")
    parser.add_argument("--dev_file", default="ft_dev.jsonl")
    parser.add_argument("--model_path", default="./qwen_translation_model")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--max_samples", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Training mode
        fine_tuner = QwenTranslationFineTuner()
        train_dataset, dev_dataset = fine_tuner.load_dataset(args.train_file, args.dev_file)
        fine_tuner.train(train_dataset, dev_dataset)
        
    elif args.mode == "translate":
        # Translation mode
        translator = QwenTranslator(args.model_path)
        
        # Load test data
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"\n🔤 Testing Translation with {args.max_samples} samples:")
        print("=" * 80)
        
        for i, item in enumerate(test_data[:args.max_samples]):
            question = item.get("question", "")
            
            print(f"\n📝 Sample {i+1}/{args.max_samples}:")
            print(f"🇬🇧 EN: {question}")
            
            try:
                translation = translator.translate_text(question)
                print(f"🇻🇳 VI: {translation}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 80)

if __name__ == "__main__":
    main()
