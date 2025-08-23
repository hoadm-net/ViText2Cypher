# � Scripts Directory

This directory contains the main evaluation scripts for the ViText2Cypher project.

## 📁 Scripts Overview

### 1. 🔄 `back_translation_analysis.py`
**Multi-metric translation quality assessment via back-translation**

- **Purpose**: Evaluates Vietnamese translation quality by back-translating to English and comparing with original
- **Methodology**: Multi-metric approach using BERT F1, ROUGE, SacreBLEU, Edit Distance
- **Output**: Quality classification (High/Medium/Low) with detailed metrics

**Key Features**:
- Enhanced multi-metric evaluation
- Statistical analysis with confidence intervals  
- Batch processing with progress tracking
- Quality threshold classification
- Comprehensive logging and error handling

**Usage Examples**:
```bash
# Basic evaluation of first 100 samples
python back_translation_analysis.py --start 0 --end 100 --enhanced

# Custom model and delay
python back_translation_analysis.py --start 500 --end 600 --model gpt-4o-mini --delay 1.0

# High confidence analysis
python back_translation_analysis.py --start 0 --end 50 --confidence-level 0.99
```

### 2. 🎯 `nl2cypher_evaluation.py`
**Enhanced cross-lingual downstream task evaluation**

- **Purpose**: Evaluates practical utility by comparing English vs Vietnamese NL2Cypher generation performance
- **Methodology**: Cross-lingual comparison with component-level F1 analysis
- **Output**: Performance metrics, consistency analysis, detailed component breakdown

**Key Features**:
- Cross-lingual comparison (English vs Vietnamese → Cypher)
- Component-level F1 analysis (MATCH, WHERE, RETURN, etc.)
- Alias consistency checking
- LangChain prompt templates
- Enhanced logging with emoji indicators
- Vietnamese language prompts for Vietnamese questions

**Usage Examples**:
```bash
# Basic cross-lingual evaluation
python nl2cypher_evaluation.py --start 0 --end 20 --delay 0.5

# Detailed analysis with debug logging
python nl2cypher_evaluation.py --start 0 --end 10 --show-details 3 --log-level DEBUG

# Large batch evaluation
python nl2cypher_evaluation.py --start 0 --end 100 --delay 1.0 --model gpt-4o-mini
```

## 📊 Key Results

### Back-translation Analysis Results
- **Quality Distribution**: 65.4% High, 31% Medium, 3.6% Low
- **BERT F1 Score**: 0.958 ± 0.025 (excellent semantic similarity)
- **Combined Score**: 0.832 ± 0.089 (strong overall quality)
- **Key Finding**: Vietnamese translations maintain high semantic fidelity

### Cross-lingual NL2Cypher Results  
- **Vietnamese Performance**: 37.6% ± 21.2%
- **English Performance**: 35.4% ± 22.2%
- **Relative Performance**: 111.8% (Vietnamese exceeds English)
- **Cross-lingual Consistency**: 40% of queries equivalent
- **Key Finding**: Vietnamese maintains downstream task performance

## 🛠️ Technical Implementation

### Script Parameters

#### Back-translation Analysis
- `--start, --end`: Sample range to evaluate
- `--model`: OpenAI model (default: from .env OPENAI_MODEL)
- `--delay`: API call delay in seconds (default: 0.5)
- `--enhanced`: Enable multi-metric analysis (recommended)
- `--confidence-level`: Statistical confidence level (default: 0.95)

#### NL2Cypher Evaluation
- `--start, --end`: Sample range to evaluate
- `--model`: OpenAI model to use
- `--delay`: API call delay in seconds
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--show-details`: Number of detailed sample results to show

## 🔧 Configuration

### Environment Variables (.env):
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini  # For NL2Cypher evaluation
```

### Common Parameters:
- `--start` / `--end` - Sample range
- `--delay` - API call delay (seconds)
- `--data-path` - Path to translated_data.json
- `--output-dir` - Output directory
- `--model` - Override model (NL2Cypher only)

## 📈 **Why Downstream Task Evaluation?**

### 🎯 **Direct Task Relevance:**
- **Translation quality ≠ Task performance**
- **End-to-end validation** of dataset utility
- **Real-world application measurement**

### 🔍 **Complementary Approaches:**
1. **Back-translation**: Linguistic fidelity
2. **NL2Cypher**: Practical effectiveness
3. **Combined**: Comprehensive quality assessment

## 📚 **Dependencies**
```bash
pip install openai python-dotenv pandas numpy nltk rouge-score bert-score sacrebleu
```

---
**Last Updated:** August 23, 2025  
**Primary Tools:** `back_translation_analysis.py` + `nl2cypher_evaluation.py`
