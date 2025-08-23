# 🎯 ViText2Cypher Evaluation Framework

Comprehensive evaluation framework for assessing Vietnamese translation quality in the ViText2Cypher dataset through multiple methodologies.

## 📚 Overview

This repository contains evaluation tools for the ViText2Cypher dataset - a Vietnamese translation of Neo4j Text2Cypher samples. The framework implements multiple evaluation approaches to assess translation quality from different perspectives.

## 🏗️ Project Structure

```
ViText2Cypher/
├── data/                           # Dataset files
│   ├── translated_data.json        # Main Vietnamese translation dataset (4,438 samples)
│   └── [other data files]
├── scripts/                        # Evaluation scripts
│   ├── back_translation_analysis.py    # Multi-metric back-translation evaluation
│   ├── nl2cypher_evaluation.py         # Enhanced cross-lingual NL2Cypher evaluation
│   └── README.md
├── templates/                      # LangChain prompt templates
│   ├── english_nl2cypher_prompt.txt
│   └── vietnamese_nl2cypher_prompt.txt
├── results/                        # Evaluation results
│   ├── back_translation/          # Back-translation analysis results
│   ├── nl2cypher_evaluation/      # Downstream task evaluation results
│   └── README.md                  # Results summary
└── README.md                      # This file
```

## � Features

### 1. 🔄 Back-translation Analysis
- **Multi-metric evaluation** using BERT F1, ROUGE, SacreBLEU, Edit Distance
- **Quality classification** (High/Medium/Low quality)
- **Statistical analysis** with confidence intervals
- **Batch processing** with progress tracking

### 2. 🎯 Enhanced NL2Cypher Cross-lingual Evaluation
- **Cross-lingual comparison** (English vs Vietnamese → Cypher generation)
- **Component-level F1 analysis** (MATCH, WHERE, RETURN, etc.)
- **Alias consistency checking**
- **LangChain prompt templates**
- **Enhanced logging** with progress tracking

## 📊 Key Results

### Back-translation Quality Assessment
- **65.4% High Quality** translations (327/500 samples)
- **BERT F1 Score**: 0.958 ± 0.025
- **Combined Score**: 0.832 ± 0.089

### Downstream Task Performance
- **Vietnamese Performance**: 37.6% ± 21.2%
- **English Performance**: 35.4% ± 22.2%
- **Translation Quality**: **EXCELLENT** (111.8% relative performance)

## 🛠️ Installation

1. **Clone repository**:
```bash
git clone <repository-url>
cd ViText2Cypher
```

2. **Set up virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install openai python-dotenv pandas langchain langchain-openai numpy nltk rouge-score sacrebleu
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🎮 Usage

### Back-translation Analysis
```bash
# Run enhanced back-translation analysis
python scripts/back_translation_analysis.py --start 0 --end 100 --enhanced

# With custom parameters
python scripts/back_translation_analysis.py --start 500 --end 600 --model gpt-4o-mini --delay 1.0
```

### Cross-lingual NL2Cypher Evaluation
```bash
# Run enhanced cross-lingual evaluation
python scripts/nl2cypher_evaluation.py --start 0 --end 20 --delay 0.5

# With detailed output
python scripts/nl2cypher_evaluation.py --start 0 --end 10 --show-details 3 --log-level DEBUG
```

## � Script Parameters

### Back-translation Analysis
- `--start, --end`: Sample range to evaluate
- `--model`: OpenAI model to use (default: from .env)
- `--delay`: Delay between API calls (seconds)
- `--enhanced`: Enable enhanced multi-metric analysis
- `--confidence-level`: Statistical confidence level (default: 0.95)

### NL2Cypher Evaluation
- `--start, --end`: Sample range to evaluate
- `--model`: OpenAI model to use
- `--delay`: Delay between API calls
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--show-details`: Number of detailed sample results to show

## 📈 Evaluation Methodologies

### 1. Multi-metric Back-translation
1. **Vietnamese → English** back-translation
2. **Similarity measurement** using multiple metrics:
   - BERT F1 Score (semantic similarity)
   - ROUGE-1, ROUGE-2, ROUGE-L (n-gram overlap)
   - SacreBLEU (translation quality)
   - Edit Distance (character-level similarity)
3. **Quality classification** based on combined scores

### 2. Cross-lingual Downstream Task Evaluation
1. **Parallel generation**: English question → Cypher, Vietnamese question → Cypher
2. **Component analysis**: F1 scores for each Cypher component
3. **Cross-lingual consistency**: Equivalence between generated queries
4. **Performance comparison**: English vs Vietnamese task performance

## 🎯 Key Findings

1. **High Translation Quality**: 65.4% of translations achieve high quality scores
2. **Semantic Preservation**: BERT F1 score of 0.958 indicates strong semantic similarity
3. **Downstream Task Equivalence**: Vietnamese performance (111.8%) matches/exceeds English
4. **Cross-lingual Consistency**: 40% of generated queries are equivalent across languages

## � Results

All evaluation results are stored in the `results/` directory with timestamped filenames. See `results/README.md` for detailed analysis of findings.
