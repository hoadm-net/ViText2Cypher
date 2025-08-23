#!/usr/bin/env python3
"""
Batch Translation Script with Range Support
Translate data from test.json with start/end parameters
"""

import json
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenBatchTranslator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned model"""
        logger.info("🔄 Loading model and tokenizer...")
        
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
            logger.info("✅ LoRA weights loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load LoRA weights: {e}")
            logger.info("Using base model instead")
    
    def translate_text(self, text: str) -> str:
        """Translate single English text to Vietnamese"""
        # Create chat template
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
    
    def batch_translate(self, input_file: str, start: int, end: int, output_file: str):
        """Batch translate with progress tracking"""
        # Load input data
        logger.info(f"📖 Loading data from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_samples = len(data)
        logger.info(f"📊 Total samples in file: {total_samples}")
        
        # Validate range
        start = max(0, start)
        end = min(total_samples, end)
        
        if start >= end:
            logger.error(f"❌ Invalid range: start={start}, end={end}")
            return
        
        subset = data[start:end]
        logger.info(f"🎯 Processing samples {start} to {end-1} ({len(subset)} samples)")
        
        # Create output structure
        results = {
            "metadata": {
                "input_file": input_file,
                "total_samples": total_samples,
                "processed_range": f"{start}-{end-1}",
                "processed_count": len(subset),
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path
            },
            "translations": []
        }
        
        # Process with progress bar
        successful = 0
        failed = 0
        
        for i, item in enumerate(tqdm(subset, desc="🔤 Translating")):
            original_question = item.get("question", "")
            schema = item.get("schema", "")
            cypher = item.get("cypher", "")
            sample_index = start + i
            
            if not original_question.strip():
                logger.warning(f"⚠️ Empty question at index {sample_index}, skipping...")
                failed += 1
                continue
            
            try:
                translation = self.translate_text(original_question)
                
                result_item = {
                    "index": sample_index,
                    "question": original_question,
                    "schema": schema,
                    "cypher": cypher,
                    "translation": translation,
                    "status": "success"
                }
                
                results["translations"].append(result_item)
                successful += 1
                
                # Log every 10th sample
                if (i + 1) % 10 == 0:
                    logger.info(f"✅ Processed {i + 1}/{len(subset)} samples")
                
            except Exception as e:
                logger.error(f"❌ Error translating sample {sample_index}: {e}")
                
                result_item = {
                    "index": sample_index,
                    "question": original_question,
                    "schema": schema,
                    "cypher": cypher,
                    "translation": "",
                    "status": "error",
                    "error": str(e)
                }
                
                results["translations"].append(result_item)
                failed += 1
        
        # Update metadata with results
        results["metadata"]["successful"] = successful
        results["metadata"]["failed"] = failed
        results["metadata"]["success_rate"] = f"{successful/len(subset)*100:.1f}%"
        
        # Save results
        logger.info(f"💾 Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Summary
        logger.info(f"🎉 Translation completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total processed: {len(subset)}")
        logger.info(f"   - Successful: {successful}")
        logger.info(f"   - Failed: {failed}")
        logger.info(f"   - Success rate: {successful/len(subset)*100:.1f}%")
        logger.info(f"   - Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch Translation with Range Support")
    parser.add_argument("--input_file", default="../data/test.json", help="Input JSON file path")
    parser.add_argument("--model_path", default="./qwen_translation_lora_trl", help="Fine-tuned model path")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate output filename
    output_file = Path(args.output_dir) / f"translated_data_{args.start}_{args.end-1}.json"
    
    # Validate inputs
    if not Path(args.input_file).exists():
        logger.error(f"❌ Input file not found: {args.input_file}")
        return
    
    if not Path(args.model_path).exists():
        logger.error(f"❌ Model path not found: {args.model_path}")
        return
    
    if args.start < 0 or args.end <= args.start:
        logger.error(f"❌ Invalid range: start={args.start}, end={args.end}")
        return
    
    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize translator and run
    logger.info(f"🚀 Starting batch translation")
    logger.info(f"📁 Input: {args.input_file}")
    logger.info(f"🤖 Model: {args.model_path}")
    logger.info(f"🎯 Range: {args.start} to {args.end-1}")
    logger.info(f"📄 Output: {output_file}")
    
    translator = QwenBatchTranslator(args.model_path)
    translator.batch_translate(args.input_file, args.start, args.end, str(output_file))

if __name__ == "__main__":
    main()
