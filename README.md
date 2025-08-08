# ViText2Cypher: Vietnamese Text-to-Cypher Dataset

A scientific research project for translating English Text2Cypher datasets to Vietnamese and building baselines for natural language to graph database query generation.

## Research Objective

This project aims to:

1. **Translate Text2Cypher Dataset**: Create a Vietnamese version of the "Text2Cypher: Bridging Natural Language and Graph Databases" dataset
2. **Build Baselines**: Develop baseline models using:
   - Prompting engineering approaches
   - Fine-tuned language models

## Current Status: Data Preprocessing Phase

This repository currently focuses on **data preprocessing** tasks. The baseline model development phase will be added in future updates.

## Project Structure

```
ViText2Cypher/
├── load_dataset.py      # Load dataset from Hugging Face
├── sample_data.py       # Randomly sample data points for translation
├── gpt_translate.py     # Translate English questions to Vietnamese
├── .env                 # Configuration for API keys and parameters
├── mint/                # Core preprocessing library
│   ├── __init__.py      # Package initialization
│   ├── dataset_handler.py   # Dataset loading and processing utilities
│   ├── sampler.py       # Data sampling utilities
│   └── translator.py    # Translation utilities with OpenAI API
├── dataset/
│   ├── train.json       # Original training data from Hugging Face
│   ├── test.json        # Original test data from Hugging Face
│   └── data.json        # Sampled data for translation
└── templates/
    └── translation_prompt.txt  # Prompt template for translation
```

## MINT Preprocessing Library

The **MINT (Machine Intelligence Translation Toolkit)** provides modular components for data preprocessing:

- **`DatasetHandler`**: Load and manage datasets from Hugging Face
- **`DataSampler`**: Sample data with reproducible random seeds
- **`CypherTranslator`**: Translate questions using OpenAI API with schema-aware prompts

## Installation

```bash
pip install datasets openai langchain python-dotenv tqdm
```

## Configuration

Create a `.env` file with your OpenAI credentials:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=500
```

## Data Preprocessing Workflow

### Step 1: Load Original Dataset

```bash
python load_dataset.py
```

Downloads the `neo4j/text2cypher-2025v1` dataset and saves to local JSON files.

### Step 2: Sample Data for Translation

```bash
# Default: sample 2000 from train.json
python sample_data.py

# Custom sampling
python sample_data.py --size 1000 --seed 42
```

Randomly samples data points from the training set for translation.

### Step 3: Translate to Vietnamese

```bash
# Translate specific range
python gpt_translate.py --start 0 --end 100

# Translate from position to end
python gpt_translate.py --start 50

# Translate entire sampled dataset
python gpt_translate.py
```

Translates English questions to Vietnamese while preserving database schema context.

### Output Format

Translated data is saved as JSON with the following structure:

```json
[
  {
    "question_en": "Which user wrote the most positive review?",
    "question_vi": "Người dùng nào đã viết đánh giá tích cực nhất?",
    "schema": "Node properties:\n- **User**\n  - `reputation`: INTEGER...",
    "cypher": "MATCH (u:User)-[:WROTE]->(r:Review)..."
  }
]
```

## Programmatic Usage

```python
from mint import DatasetHandler, DataSampler, CypherTranslator

# Load original dataset
handler = DatasetHandler()
data = handler.load_from_huggingface("neo4j/text2cypher-2025v1")

# Sample data for translation
sampler = DataSampler(random_seed=42)
sampled_data = sampler.sample_data(data['train'], 1000)

# Translate to Vietnamese
translator = CypherTranslator()
translated = translator.translate_batch(sampled_data)
```

## Complete Preprocessing Pipeline

```bash
# 1. Load original dataset
python load_dataset.py

# 2. Sample data points for translation
python sample_data.py --size 2000 --seed 42

# 3. Translate sampled data
python gpt_translate.py --start 0 --end 2000
```

## Research Roadmap

### ✅ Phase 1: Data Preprocessing (Current)
- [x] Dataset loading from Hugging Face
- [x] Data sampling utilities
- [x] English to Vietnamese translation
- [x] Schema-aware translation prompts

### 🔄 Phase 2: Baseline Development (Upcoming)
- [ ] Prompting engineering baselines
- [ ] Few-shot learning approaches
- [ ] Fine-tuned model baselines
- [ ] Evaluation metrics and benchmarks

### 🔄 Phase 3: Model Evaluation (Future)
- [ ] Performance comparison
- [ ] Error analysis
- [ ] Dataset quality assessment

## Contributing

This is a research project. For questions about the methodology or to contribute to the baseline development, please refer to the research documentation (to be added in Phase 2).
