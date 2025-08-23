# 📊 ViText2Cypher Evaluation Framework

This document describes the evaluation tools and methodologies used in the ViText2Cypher project for assessing cross-lingual natural language to Cypher query generation.

## 🎯 Overview

The evaluation framework consists of two main components:
1. **Back Translation Analysis** - Multi-metric translation quality assessment with comprehensive statistical analysis
2. **NL2Cypher Evaluation** - Advanced cross-lingual Cypher generation evaluation with component-wise analysis

Both tools feature **unified output structure** with standardized folder organization and dual-file reporting (overview + details).

---

## 🔄 Back Translation Analysis

### Purpose
Evaluates the quality of bidirectional translation between English and Vietnamese natural language questions using comprehensive multi-metric analysis.

### Features
- **Multi-Metric Evaluation**: ROUGE, BERT Score, Semantic Similarity, Edit Distance, BLEU, Word F1, Jaccard
- **Combined Scoring**: Weighted composite score for overall translation quality assessment
- **Statistical Analysis**: Mean, median, standard deviation, min/max for all metrics
- **Quality Classification**: Automatic categorization (Cao/Trung bình/Thấp) based on combined scores
- **Detailed Reporting**: Sample-by-sample breakdown with metric explanations

### Key Metrics

#### Core Translation Metrics
- **ROUGE (1, 2, L)**: N-gram overlap and longest common subsequence similarity
- **BERT F1/Precision/Recall**: Contextual embedding-based semantic similarity
- **Semantic Similarity**: Sentence transformer cosine similarity
- **Edit Distance Ratio**: Character-level string similarity (1 - normalized edit distance)
- **Word F1**: Token-level precision and recall
- **SacreBLEU**: Improved BLEU implementation with better preprocessing
- **Jaccard Similarity**: Set-based token overlap coefficient

#### Composite Scoring
**Combined Score Formula** (weighted average):
- Semantic Similarity: 30%
- BERT F1: 25% 
- ROUGE-1 F1: 15%
- ROUGE-L F1: 15%
- Word F1: 10%
- Edit Distance: 10%
- SacreBLEU: 5%

**Quality Thresholds**:
- 🟢 **Cao** (≥0.80): High-quality translation
- 🟡 **Trung bình** (≥0.65): Acceptable translation quality  
- 🔴 **Thấp** (<0.65): Poor translation quality

#### Metric Comparison Table
| Metric | Purpose | Strength | Typical Range |
|--------|---------|----------|---------------|
| **Semantic Similarity** | Meaning preservation | Best for semantic equivalence | 85-99% |
| **BERT F1** | Contextual understanding | Most reliable/stable | 90-99% |
| **ROUGE-1** | Word overlap | Good for lexical similarity | 70-95% |
| **ROUGE-L** | Sequence similarity | Captures word order | 70-95% |
| **Edit Distance** | Character similarity | Detects typos/minor changes | 80-98% |
| **Word F1** | Token-level accuracy | Precise but sensitive | 60-90% |
| **SacreBLEU** | Translation quality | Standard for MT evaluation | 10-80% |

### Recent Performance (30 samples)
```
Combined Score: 84.2% (σ=9.3%)
├── Semantic Similarity: 92.1% (best performing)
├── BERT F1: 95.8% (most reliable)
├── ROUGE-1 F1: 82.1% (good n-gram overlap)
└── Word F1: 71.6% (token-level accuracy)

Quality Distribution:
├── 🟢 Cao: 26 samples (86.7%)
├── 🟡 Trung bình: 4 samples (13.3%)
└── 🔴 Thấp: 0 samples (0.0%)
```

### Usage

```bash
# Basic back translation analysis (recommended)
python scripts/back_translation_analysis.py --start 0 --end 30

# Large-scale evaluation
python scripts/back_translation_analysis.py --start 0 --end 100

# Quick validation
python scripts/back_translation_analysis.py --start 0 --end 5

# Custom parameters
python scripts/back_translation_analysis.py \
  --start 0 \
  --end 500 \
  --delay 1.0 \
  --log-level INFO
```

### Output Structure
```
results/back_translation_analysis/
└── back_translation_analysis_{start}_{end}/
    ├── overview.txt          # Formatted metrics table & sample breakdown
    └── details.json          # Complete evaluation data with all metrics
```

#### Sample Overview Report
```
🎯 BACK TRANSLATION ANALYSIS REPORT
================================================================================
📋 Sample Range: 0 - 30 (total: 30)
🏆 TRANSLATION QUALITY METRICS
┌─────────────────────────┬─────────────┬─────────────┬─────────────┐
│ Metric                  │    Mean     │   Median    │    Std      │
├─────────────────────────┼─────────────┼─────────────┼─────────────┤
│ Combined Score          │     0.842   │     0.839   │     0.093   │
│ Semantic Similarity     │     0.921   │     0.920   │     0.055   │
│ BERT F1                 │     0.958   │     0.958   │     0.029   │
└─────────────────────────┴─────────────┴─────────────┴─────────────┘
🏆 Overall Translation Quality: 🟡 GOOD (84.2%)
```

---

## 🚀 NL2Cypher Cross-Lingual Evaluation

### Purpose
Comprehensive evaluation framework for assessing the quality of Cypher query generation from natural language in both English and Vietnamese.

### Features
- **Cross-Lingual Comparison**: English vs Vietnamese vs Ground Truth analysis
- **Component-wise Analysis**: Detailed breakdown of Cypher query components
- **Syntax Validation**: Advanced local syntax checking without external dependencies
- **Semantic Normalization**: Alias-aware component comparison for accurate F1 scoring
- **Cost Optimization**: Only 2 API calls per sample (50% reduction from baseline)

### Evaluation Dimensions

#### 1. **Exact Match (EM)**
Binary metric indicating perfect query match:
```
EM = 1 if predicted_query == ground_truth_query else 0
```

#### 2. **Component F1 Score**
Precision and recall-based evaluation of Cypher components:
```
Precision = |predicted ∩ ground_truth| / |predicted|
Recall = |predicted ∩ ground_truth| / |ground_truth|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 3. **Categorized F1 Analysis**
Components grouped into logical categories:
- **MATCH**: Node and relationship patterns
- **WHERE**: Filtering conditions and constraints
- **RETURN**: Output specifications and projections
- **KEYWORDS**: Other Cypher keywords (WITH, ORDER BY, LIMIT, etc.)

#### 4. **Translation Quality Assessment**
Relative performance metric:
```
Vietnamese Relative F1 = Vietnamese_F1 / English_F1
```

Quality Scale:
- 🟢 **EXCELLENT** (≥95%): Near-perfect translation quality
- 🟡 **GOOD** (≥85%): High-quality translation with minor differences  
- 🟠 **FAIR** (≥70%): Acceptable translation quality
- 🔴 **POOR** (<70%): Translation quality needs improvement

### Common Command-Line Parameters
Both evaluation scripts support:
- `--start`: Starting index for evaluation range
- `--end`: Ending index for evaluation range  
- `--delay`: Delay between API calls (default: 1.0s)
- `--log-level`: Logging verbosity (INFO, DEBUG, WARNING)

### Advanced Features

#### Semantic Normalization
- **Alias Consistency**: Handles variable name differences (e.g., `n` vs `t`)
- **Component Ordering**: Order-independent comparison for equivalent queries
- **Property Normalization**: Case-insensitive property and label matching

#### Enhanced Syntax Validation
- **Local Validation**: No external Neo4j dependency required
- **Pattern Recognition**: Validates Cypher syntax using regex patterns
- **Error Reporting**: Detailed syntax error explanations

### Usage

```bash
# Basic evaluation (recommended)
python scripts/nl2cypher_evaluation.py --start 0 --end 50

# Large-scale evaluation
python scripts/nl2cypher_evaluation.py --start 0 --end 1000

# Quick validation
python scripts/nl2cypher_evaluation.py --start 0 --end 5

# Custom parameters
python scripts/nl2cypher_evaluation.py \
  --start 0 \
  --end 100 \
  --delay 1.0 \
  --log-level INFO
```

### Output Structure
```
results/nl2cypher_evaluation/
└── nl2cypher_evaluation_{start}_{end}/
    ├── overview.txt          # Performance summary & analysis
    └── details.json          # Complete evaluation results
```

#### Overview Report Contents
- **Performance Metrics**: EM and F1 scores for English/Vietnamese
- **Cross-lingual Similarity**: English vs Vietnamese comparison
- **Translation Quality**: Automated quality assessment
- **Component Breakdown**: F1 scores by category
- **Sample Analysis**: Detailed breakdown of individual samples

#### Details JSON Contents
- **Metadata**: Evaluation parameters and dataset information
- **Statistics**: Aggregated metrics across all samples
- **Individual Results**: Complete evaluation data per sample
- **Component Analysis**: Detailed F1 scores for each Cypher component

---

## 🛠️ Technical Requirements

### Dependencies
```bash
# Core dependencies
pip install openai langchain numpy pandas

# For back translation analysis (multi-metric evaluation)
pip install sentence-transformers sacrebleu rouge-score bert-score

# For NL2Cypher evaluation  
pip install regex pathlib

# Optional: for enhanced text processing
pip install nltk spacy
```

### Environment Setup
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Model Downloads
Some tools require pre-trained models:
```bash
# For BERT Score (automatic download on first use)
python -c "import bert_score; bert_score.score(['test'], ['test'], lang='en')"

# For sentence transformers (automatic download)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## 📝 Best Practices

### Evaluation Guidelines
1. **Sample Size**: Use at least 30-50 samples for reliable statistics
2. **API Rate Limits**: Use appropriate delays to respect OpenAI rate limits
3. **Result Validation**: Review sample outputs manually for quality assurance
4. **Statistical Significance**: Consider confidence intervals for result interpretation
5. **Progressive Testing**: Start with small samples (3-5) before scaling up

### Performance Optimization
1. **Batch Processing**: Evaluate in chunks for large datasets
2. **Cost Management**: Monitor API usage with built-in cost optimization
3. **Local Validation**: Leverage local syntax validation to reduce API calls
4. **Result Caching**: Reuse evaluation results where possible
5. **Multi-Metric Analysis**: Combine multiple metrics for robust quality assessment

### Quality Assurance
1. **Manual Review**: Inspect sample outputs for evaluation accuracy
2. **Cross-validation**: Compare results across different evaluation runs
3. **Component Analysis**: Use categorized F1 scores for detailed insights
4. **Translation Assessment**: Monitor relative F1 for translation quality trends
5. **Metric Interpretation**: Understand strengths/weaknesses of each metric

### Recent Improvements
- ✅ **Clean Architecture**: Removed all "enhanced_" prefixes for cleaner naming
- ✅ **Unified Output**: Standardized folder structure across both tools
- ✅ **Multi-Metric Evaluation**: Comprehensive translation quality assessment
- ✅ **Robust Statistics**: Mean, median, std dev for all metrics
- ✅ **Quality Classification**: Automatic assessment with clear thresholds

---

## 🔄 Migration Guide: Output Structure Changes

### Before (Enhanced Architecture)
```
results/
├── enhanced_back_translation_analysis/
│   └── enhanced_back_translation_analysis_{start}_{end}/
└── enhanced_nl2cypher_evaluation/
    └── enhanced_nl2cypher_evaluation_{start}_{end}/
```

### After (Clean Architecture) 
```
results/
├── back_translation_analysis/
│   └── back_translation_analysis_{start}_{end}/
│       ├── overview.txt      # Formatted metrics table
│       └── details.json      # Complete evaluation data
└── nl2cypher_evaluation/
    └── nl2cypher_evaluation_{start}_{end}/
        ├── overview.txt      # Performance summary
        └── details.json      # Detailed results
```

### Benefits of New Structure
- **Cleaner naming**: Removed redundant "enhanced_" prefixes
- **Consistent organization**: Same folder pattern for both tools
- **Dual-file system**: Overview for humans, details for machines
- **Better discoverability**: Clear tool names without prefixes

---

## 📋 Summary

The ViText2Cypher evaluation framework provides comprehensive tools for assessing cross-lingual NL2Cypher generation quality. 

### Key Features
- **Unified Architecture**: Clean naming and consistent output structure across tools
- **Multi-Metric Analysis**: Comprehensive translation quality assessment with 7+ metrics
- **Statistical Rigor**: Mean, median, standard deviation analysis for reliable results
- **Quality Classification**: Automatic assessment with clear performance thresholds
- **Cost Optimization**: Efficient API usage with local validation where possible
- **Detailed Reporting**: Both human-readable overviews and machine-readable detailed results

### Typical Workflow
1. **Quick Validation**: Test with 3-5 samples to ensure syntax correctness
2. **Development Testing**: Use 30-50 samples for reliable statistics  
3. **Production Evaluation**: Scale up to 100+ samples for final assessment
4. **Result Analysis**: Review both overview reports and detailed JSON data

### Performance Benchmarks (30-sample evaluation)
- **Back Translation Quality**: 84.2% combined score (Good quality)
- **Translation Consistency**: 86.7% samples achieve "Cao" quality level
- **Metric Reliability**: BERT F1 (95.8%) and Semantic Similarity (92.1%) most stable
- **Processing Efficiency**: ~2 API calls per sample with local validation

For implementation details and code structure, refer to the main project documentation and source code comments.
