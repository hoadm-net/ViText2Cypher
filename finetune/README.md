# 🔧 Fine-tuning Pipeline

Fine-tune Qwen2.5-7B-Instruct for English to Vietnamese translation task using LoRA (Low-Rank Adaptation).

## 📝 Overview

This pipeline fine-tunes a large language model to translate English natural language questions to Vietnamese for the NL2Cypher task. It uses the augmented data from the data augmentation pipeline as training material.

### 🔄 Workflow

```
augmented_data.json (4k samples)
           ↓
   split_finetune_data.py (80:20 split)
           ↓
   ft_train.jsonl + ft_dev.jsonl
           ↓
   fine_tune_qwen_trl.py (LoRA training)
           ↓
   Fine-tuned Translation Model
```

## 🛠️ Scripts Overview

### Core Scripts

1. **`split_finetune_data.py`** - Prepare training data
   - Splits `augmented_data.json` into train/dev sets (80:20)
   - Converts to chat message format for fine-tuning
   - Outputs: `ft_train.jsonl`, `ft_dev.jsonl`

2. **`fine_tune_qwen_trl.py`** - Main fine-tuning script
   - Fine-tunes Qwen2.5-7B-Instruct with LoRA
   - Uses TRL SFTTrainer for efficient training
   - Supports 4-bit quantization for GPU efficiency

3. **`translate.py`** - Batch translation with fine-tuned model
   - Uses the fine-tuned model for translation tasks
   - Supports range-based processing

### Utility Scripts

4. **`merge_translations.py`** - Merge translation results
   - Combines multiple translation output files
   - Organizes data with proper indexing

5. **`qwen_template.py`** - Template fine-tuning script
   - Alternative fine-tuning implementation
   - Reference implementation

6. **`push_to_hf.py`** - Upload model to Hugging Face Hub
   - Publishes fine-tuned model
   - Creates model cards and documentation

## 🚀 Usage

### Step 1: Prepare Training Data
```bash
cd finetune
python split_finetune_data.py
```

**Input**: `../data/augmented_data.json` (4k samples)
**Output**: 
- `../data/ft_train.jsonl` (~3.2k samples)  
- `../data/ft_dev.jsonl` (~0.8k samples)

### Step 2: Fine-tune Model
```bash
python fine_tune_qwen_trl.py
```

**Requirements**:
- GPU with 8GB+ VRAM (RTX 3090 recommended)
- ~2-3 hours training time
- ~$0 cost (local training)

**Output**: 
- `./qwen_translation_lora_trl/` (LoRA weights)
- Training logs and checkpoints

### Step 3: Translate with Fine-tuned Model
```bash
python translate.py --model_path ./qwen_translation_lora_trl --start 0 --end 100
```

**Input**: `../data/test.json`
**Output**: `translated_data_0_99.json`

## 🔧 Configuration

### Environment Setup
```bash
# Install dependencies
pip install torch transformers trl peft datasets bitsandbytes

# For GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Hardware Requirements
- **Minimum**: 8GB GPU VRAM (RTX 3090)
- **Recommended**: 16GB+ GPU VRAM (RTX 4090, V100)
- **CPU**: 16GB+ RAM
- **Storage**: 20GB+ for model and checkpoints

### Training Parameters
```python
# LoRA Configuration
r=16                    # Rank
lora_alpha=32          # Scaling factor
target_modules=[...]   # Attention layers
dropout=0.1            # Regularization

# Training Configuration
batch_size=2           # Per device
learning_rate=3e-4     # Optimized for LoRA
epochs=3               # Prevents overfitting
```

## 📊 Data Format

### Input Format (augmented_data.json)
```json
{
  "question": "What are the names of all suppliers?",
  "schema": "Node properties: Supplier...",
  "cypher": "MATCH (s:Supplier) RETURN s.name",
  "translation": "Tên của tất cả các nhà cung cấp là gì?"
}
```

### Training Format (ft_train.jsonl)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional translator..."
    },
    {
      "role": "user", 
      "content": "<QUESTION>What are the names...</QUESTION>\n<SCHEMA>Node properties...</SCHEMA>"
    },
    {
      "role": "assistant",
      "content": "Tên của tất cả các nhà cung cấp là gì?"
    }
  ]
}
```

## 💡 Optimizations

### Performance
- **LoRA Efficiency**: Only train 0.1% of parameters
- **4-bit Quantization**: Reduce memory usage by 75%
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: BF16 for stability

### Quality Control
- **Validation Split**: 20% for monitoring overfitting
- **Early Stopping**: Prevent overtraining
- **Best Model Selection**: Save best checkpoint
- **Evaluation Metrics**: Track translation quality

## 📈 Expected Results

### Performance Metrics
- **Training Time**: 2-3 hours on RTX 3090
- **Memory Usage**: ~6-8GB VRAM with 4-bit quantization
- **Model Size**: ~50MB LoRA weights (vs 14GB full model)

### Translation Quality
- **BLEU Score**: 85-90% (estimated)
- **Semantic Similarity**: 90-95%
- **Fluency**: High (thanks to Qwen2.5 base model)
- **Domain Adaptation**: Specialized for database queries

## 🚨 Important Notes

1. **GPU Requirements**: Minimum 8GB VRAM required
2. **Training Stability**: Use BF16 instead of FP16 for Qwen models
3. **Memory Management**: Enable gradient checkpointing if VRAM limited
4. **Quality Monitoring**: Check validation loss to prevent overfitting
5. **Model Saving**: LoRA weights are much smaller than full models

## 🔗 Integration

The fine-tuned model can be used in:
- **Evaluation Pipeline**: Replace GPT-4 translation for cost savings
- **Production Systems**: Fast local translation inference
- **Further Training**: Base for domain-specific adaptations
- **Model Serving**: Deploy with vLLM or similar frameworks

---

**🎯 Goal**: Create a specialized English→Vietnamese translation model for database query contexts, reducing API costs while maintaining high quality.
