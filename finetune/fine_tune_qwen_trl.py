#!/usr/bin/env python3
"""
Fine-tuning Qwen2.5-7B-Instruct for English to Vietnamese translation task
using TRL SFTTrainer with LoRA (Updated 2025)
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)    # Main execution
    train_dataset, dev_dataset = fine_tuner.load_dataset(
        train_file="../data/ft_train.jsonl",
        dev_file="../data/ft_dev.jsonl"
    )
    
    # Training
    trainer = fine_tuner.train(train_dataset, dev_dataset)
    
    # Evaluation
    fine_tuner.evaluate_model(
        trainer=trainer,
        test_file="../data/test.json",
        output_file="translation_results_trl.json",
        max_samples=10
    ) SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset
import os
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTranslationFineTunerTRL:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.output_dir = "./qwen_translation_lora_trl"
        
    def load_dataset(self, train_file: str, dev_file: str) -> tuple:
        """Load and format data for TRL SFTTrainer"""
        logger.info("Loading data...")
        
        def load_jsonl(file_path: str) -> List[Dict]:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        # Load data
        train_data = load_jsonl(train_file)
        dev_data = load_jsonl(dev_file)
        
        logger.info(f"Loaded {len(train_data)} training samples and {len(dev_data)} validation samples")
        
        # TRL SFTTrainer automatically handles format, no preprocessing needed
        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)
        
        return train_dataset, dev_dataset
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration with latest best practices"""
        logger.info("Creating LoRA configuration...")
        
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,  # scaling factor
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,  # Rank-Stabilized LoRA - latest!
            # Can add if need to fine-tune embedding:
            # modules_to_save=["embed_tokens", "lm_head"]
        )
        
        logger.info("LoRA config created with RSLoRA enabled")
        return lora_config
    
    def create_sft_config(self) -> SFTConfig:
        """Create SFT configuration with latest optimization"""
        logger.info("Creating SFT configuration...")
        
        sft_config = SFTConfig(
            # Output settings
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Training parameters - Optimized for RTX 3090
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Increase from 1 -> 2
            per_device_eval_batch_size=2,   # Increase from 1 -> 2  
            gradient_accumulation_steps=4,  # Decrease from 8 -> 4 (keep effective batch = 8)
            learning_rate=3e-4,  # Increase from 2e-4 -> 3e-4 for faster convergence
            weight_decay=0.01,
            warmup_steps=100,
            
            # Optimization settings
            bf16=True,  # bfloat16 for stability
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            
            # Evaluation settings - Optimized for speed
            eval_strategy="steps",
            eval_steps=200,  # Increase from 100 -> 200 (less eval)
            save_steps=400,  # Increase from 200 -> 400
            save_total_limit=2,  # Decrease from 3 -> 2
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_steps=10,
            report_to=None,
            
            # TRL specific settings
            max_length=2048,
            eos_token="<|im_end|>",  # Important for Qwen!
            assistant_only_loss=False,  # Disable because Qwen chat template doesn't support
            packing=False,  # Disable to avoid Flash Attention requirement
            padding_free=False,  # Disable to avoid attention issues
            
            # Model initialization - Force GPU usage
            model_init_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": {"": 0},  # Force GPU 0 instead of "auto"
                "trust_remote_code": True,
                "attn_implementation": "eager",  # Use eager attention
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            }
        )
        
        logger.info("SFT config created with stable configuration (packing disabled for compatibility)")
        return sft_config
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Training function with TRL SFTTrainer"""
        logger.info("Starting training with TRL SFTTrainer...")
        
        # Create configurations
        lora_config = self.create_lora_config()
        sft_config = self.create_sft_config()
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model_name,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
        )
        
        # Display trainable parameters info
        if hasattr(trainer.model, 'print_trainable_parameters'):
            trainer.model.print_trainable_parameters()
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        
        # Save tokenizer - Fix deprecation warning
        tokenizer = trainer.processing_class if hasattr(trainer, 'processing_class') else trainer.tokenizer
        if tokenizer:
            tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed! Model saved at {self.output_dir}")
        
        return trainer
    
    def evaluate_model(self, trainer: SFTTrainer, test_file: str, output_file: str, max_samples: int = 10):
        """Evaluate model on test set"""
        logger.info(f"Evaluating model on {max_samples} test samples...")
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        
        for i, item in enumerate(test_data[:max_samples]):
            question = item.get("question", "")
            
            # Create messages format for chat template
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional translator. Only return the Vietnamese translation of the following question. Keep technical keywords and proper names unchanged."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            # Apply chat template - Fix deprecation warning
            tokenizer = trainer.processing_class if hasattr(trainer, 'processing_class') else trainer.tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
            
            # Generate - Fix dtype issues với comprehensive handling
            with torch.no_grad():
                # Ensure model is in eval mode
                trainer.model.eval()
                
                try:
                    # First attempt with autocast disabled
                    with torch.cuda.amp.autocast(enabled=False):
                        # Convert model to float32 for generation if needed
                        original_dtype = next(trainer.model.parameters()).dtype
                        if original_dtype == torch.bfloat16:
                            trainer.model = trainer.model.float()
                        
                        outputs = trainer.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.3,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )
                        
                        # Convert back to original dtype
                        if original_dtype == torch.bfloat16:
                            trainer.model = trainer.model.bfloat16()
                            
                except Exception as e:
                    logger.warning(f"Generation failed: {e}. Skipping this sample...")
                    continue
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "<|im_start|>assistant" in response:
                translation = response.split("<|im_start|>assistant\n")[-1].strip()
            else:
                translation = response[len(prompt):].strip()
            
            results.append({
                "original": question,
                "translation": translation
            })
            
            logger.info(f"Sample {i+1}/{max_samples}")
            logger.info(f"Original: {question[:100]}...")
            logger.info(f"Translation: {translation}")
            logger.info("-" * 80)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation completed! Results saved at {output_file}")

def main():
    """Main function"""
    logger.info("🚀 Starting Fine-tuning Qwen2.5 with TRL SFTTrainer")
    
    # Initialize fine-tuner
    fine_tuner = QwenTranslationFineTunerTRL()
    
    # Load data
    train_dataset, dev_dataset = fine_tuner.load_dataset(
        train_file="../data/ft_train.jsonl",
        dev_file="../data/ft_dev.jsonl"
    )
    
    # Training
    trainer = fine_tuner.train(train_dataset, dev_dataset)
    
    # Evaluation
    fine_tuner.evaluate_model(
        trainer=trainer,
        test_file="../data/test.json",
        output_file="translation_results_trl.json",
        max_samples=10
    )
    
    logger.info("✅ Completed everything! Model has been fine-tuned and evaluated.")

if __name__ == "__main__":
    main()
