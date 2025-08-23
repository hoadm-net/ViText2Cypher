#!/usr/bin/env python3
"""
🚀 Back-Translation Analysis for ViText2Cypher Dataset
====================================================

DESCRIPTION:
    Advanced evaluation framework using multiple metrics for robust assessment 
    of Vietnamese translation quality via back-translation consistency.

FEATURES:
    ✅ BERT F1 Score - Semantic similarity using BERT embeddings
    ✅ ROUGE-1/2/L - Flexible n-gram overlap metrics  
    ✅ Semantic Similarity - OpenAI embeddings cosine similarity
    ✅ Edit Distance Ratio - Character-level similarity
    ✅ SacreBLEU - Improved BLEU implementation
    ✅ Word F1 - Word-level precision/recall
    ✅ Ensemble Scoring - Weighted combination of all metrics

USAGE:
    python scripts/back_translation_analysis.py --start 0 --end 100 --delay 1.0

AUTHOR: Generated for ViText2Cypher evaluation
DATE: August 2025
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import sys
import os
import re

# Thêm root directory vào Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import nltk
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    from sacrebleu import sentence_bleu
    import difflib
except ImportError:
    print("Cần cài đặt required libraries:")
    print("pip install openai python-dotenv nltk rouge-score bert-score sacrebleu")
    sys.exit(1)

# Load environment variables từ .env file
load_dotenv()

# Download NLTK data nếu cần
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class BackTranslationAnalyzer:
    def __init__(self, api_key=None):
        """Khởi tạo analyzer với OpenAI client và multiple metrics"""
        self.client = OpenAI(api_key=api_key)
        # Lấy model từ environment variables
        self.translation_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Prompt template cho back-translation
        self.back_translation_prompt = """Translate the following Vietnamese text to English. 
Be precise and maintain the original meaning as much as possible.

Vietnamese text: {vietnamese_text}

English translation:"""
    
    def back_translate(self, vietnamese_text):
        """Dịch ngược từ tiếng Việt sang tiếng Anh"""
        try:
            prompt = self.back_translation_prompt.format(vietnamese_text=vietnamese_text)
            
            response = self.client.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Lỗi khi back-translate: {e}")
            return None
    
    def get_embedding(self, text):
        """Lấy embedding vector cho text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Lỗi khi lấy embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Tính cosine similarity giữa hai vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def calculate_rouge_scores(self, reference, candidate):
        """Tính ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall,
            }
        except Exception as e:
            print(f"Lỗi khi tính ROUGE scores: {e}")
            return {key: 0.0 for key in ['rouge1_f', 'rouge1_p', 'rouge1_r', 
                                       'rouge2_f', 'rouge2_p', 'rouge2_r',
                                       'rougeL_f', 'rougeL_p', 'rougeL_r']}
    
    def calculate_bert_score(self, reference, candidate):
        """Tính BERTScore (semantic similarity using BERT)"""
        try:
            # BERTScore sử dụng BERT để đo semantic similarity
            P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
            return {
                'bert_precision': float(P[0]),
                'bert_recall': float(R[0]),
                'bert_f1': float(F1[0])
            }
        except Exception as e:
            print(f"Lỗi khi tính BERTScore: {e}")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def calculate_edit_distance_ratio(self, reference, candidate):
        """Tính Edit Distance Ratio (Levenshtein-based similarity)"""
        try:
            # Sequence similarity dựa trên edit distance
            similarity = difflib.SequenceMatcher(None, reference.lower(), candidate.lower()).ratio()
            return similarity
        except Exception as e:
            print(f"Lỗi khi tính Edit Distance Ratio: {e}")
            return 0.0
    
    def calculate_sacrebleu(self, reference, candidate):
        """Tính SacreBLEU (improved BLEU implementation)"""
        try:
            # SacreBLEU có tokenization tốt hơn
            bleu = sentence_bleu(candidate, [reference])
            return bleu.score / 100.0  # Convert to 0-1 range
        except Exception as e:
            print(f"Lỗi khi tính SacreBLEU: {e}")
            return 0.0
    
    def calculate_word_overlap_metrics(self, reference, candidate):
        """Tính các metrics dựa trên word overlap"""
        try:
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            intersection = ref_words.intersection(cand_words)
            union = ref_words.union(cand_words)
            
            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Overlap coefficient  
            overlap = len(intersection) / min(len(ref_words), len(cand_words)) if min(len(ref_words), len(cand_words)) > 0 else 0.0
            
            # Word-level precision and recall
            precision = len(intersection) / len(cand_words) if cand_words else 0.0
            recall = len(intersection) / len(ref_words) if ref_words else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'jaccard_similarity': jaccard,
                'overlap_coefficient': overlap,
                'word_precision': precision,
                'word_recall': recall,
                'word_f1': f1
            }
        except Exception as e:
            print(f"Lỗi khi tính Word Overlap metrics: {e}")
            return {
                'jaccard_similarity': 0.0,
                'overlap_coefficient': 0.0,
                'word_precision': 0.0,
                'word_recall': 0.0,
                'word_f1': 0.0
            }
    
    def classify_consistency_enhanced(self, metrics):
        """Phân loại consistency dựa trên ensemble của multiple metrics"""
        # Weighted combination của các metrics
        weights = {
            'semantic_similarity': 0.25,    # Semantic meaning
            'bert_f1': 0.20,               # BERT-based semantic
            'rouge1_f': 0.15,              # Unigram overlap
            'rougeL_f': 0.15,              # Longest common subsequence
            'word_f1': 0.10,               # Word-level F1
            'edit_distance_ratio': 0.10,    # Character-level similarity
            'sacrebleu': 0.05              # Improved BLEU
        }
        
        # Calculate weighted score
        combined_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                combined_score += weight * metrics[metric]
                total_weight += weight
        
        if total_weight > 0:
            combined_score = combined_score / total_weight
        
        # Classification thresholds
        if combined_score >= 0.80:
            return "Cao", combined_score
        elif combined_score >= 0.65:
            return "Trung bình", combined_score
        else:
            return "Thấp", combined_score
    
    def analyze_sample(self, sample, index):
        """Phân tích một sample với multiple metrics"""
        original_english = sample.get('question', '')
        vietnamese_text = sample.get('translation', '')
        
        if not original_english or not vietnamese_text:
            return {
                'index': index,
                'error': 'Thiếu text tiếng Anh hoặc tiếng Việt',
                'metrics': None
            }
        
        # Step 1: Back-translate Vietnamese to English
        back_translated = self.back_translate(vietnamese_text)
        if back_translated is None:
            return {
                'index': index,
                'error': 'Không thể back-translate',
                'metrics': None
            }
        
        # Step 2: Calculate all metrics
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(original_english, back_translated)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = self.calculate_bert_score(original_english, back_translated)
        metrics.update(bert_scores)
        
        # Edit distance ratio
        metrics['edit_distance_ratio'] = self.calculate_edit_distance_ratio(original_english, back_translated)
        
        # SacreBLEU
        metrics['sacrebleu'] = self.calculate_sacrebleu(original_english, back_translated)
        
        # Word overlap metrics
        word_metrics = self.calculate_word_overlap_metrics(original_english, back_translated)
        metrics.update(word_metrics)
        
        # Semantic similarity (OpenAI embeddings)
        original_embedding = self.get_embedding(original_english)
        back_embedding = self.get_embedding(back_translated)
        
        if original_embedding is not None and back_embedding is not None:
            metrics['semantic_similarity'] = self.cosine_similarity(original_embedding, back_embedding)
        else:
            metrics['semantic_similarity'] = None
        
        # Step 3: Enhanced classification
        consistency_level, combined_score = self.classify_consistency_enhanced(metrics)
        
        return {
            'index': index,
            'original_english': original_english,
            'vietnamese_text': vietnamese_text,
            'back_translation': back_translated,
            'metrics': metrics,
            'combined_score': combined_score,
            'consistency_level': consistency_level,
            'error': None
        }

def load_data(data_path):
    """Load dữ liệu từ translated_data.json"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Đã load {len(data)} samples từ {data_path}")
        return data
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}")
        return None

def calculate_statistics(results):
    """Tính toán thống kê từ kết quả với multiple metrics"""
    valid_results = [r for r in results if r['metrics'] is not None]
    
    if not valid_results:
        return {
            'total_samples': len(results),
            'valid_samples': 0,
            'error_samples': len(results),
            'error_rate': 1.0
        }
    
    # Collect all metric values
    metric_names = [
        'rouge1_f', 'rouge2_f', 'rougeL_f',
        'bert_f1', 'bert_precision', 'bert_recall',
        'edit_distance_ratio', 'sacrebleu',
        'jaccard_similarity', 'word_f1',
        'semantic_similarity', 'combined_score'
    ]
    
    stats = {
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'error_samples': len(results) - len(valid_results),
        'error_rate': (len(results) - len(valid_results)) / len(results),
    }
    
    # Calculate statistics for each metric
    for metric in metric_names:
        values = []
        for result in valid_results:
            if result['metrics'] and metric in result['metrics'] and result['metrics'][metric] is not None:
                values.append(result['metrics'][metric])
            elif metric == 'combined_score' and result.get('combined_score') is not None:
                values.append(result['combined_score'])
        
        if values:
            stats[f'{metric}_mean'] = float(np.mean(values))
            stats[f'{metric}_median'] = float(np.median(values))
            stats[f'{metric}_std'] = float(np.std(values))
            stats[f'{metric}_min'] = float(np.min(values))
            stats[f'{metric}_max'] = float(np.max(values))
    
    # Consistency distribution
    consistency_counts = {}
    for result in valid_results:
        level = result['consistency_level']
        consistency_counts[level] = consistency_counts.get(level, 0) + 1
    
    consistency_distribution = {}
    for level, count in consistency_counts.items():
        consistency_distribution[level] = {
            'count': count,
            'percentage': (count / len(valid_results)) * 100
        }
    
    stats['consistency_distribution'] = consistency_distribution
    
    return stats

def save_results(results, stats, folder_path, metadata):
    """Lưu kết quả vào folder với overview.txt và details.json"""
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON data
    details_data = {
        'metadata': metadata,
        'statistics': stats,
        'results': results
    }
    
    details_path = folder_path / 'details.json'
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(details_data, f, ensure_ascii=False, indent=2)
    
    # Generate readable overview report
    overview_path = folder_path / 'overview.txt'
    generate_readable_report(results, stats, metadata, overview_path)
    
    print(f"💾 Đã lưu debug results vào: {details_path}")
    print(f"📊 Đã lưu báo cáo TXT vào: {overview_path}")

def generate_readable_report(results, stats, metadata, output_path):
    """Tạo báo cáo readable cho back translation analysis"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("🎯 BACK TRANSLATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Metadata section
        f.write("📋 EVALUATION METADATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Timestamp:        {metadata['timestamp']}\n")
        f.write(f"Data Path:        {metadata['data_path']}\n")
        f.write(f"Sample Range:     {metadata['start_index']} - {metadata['end_index']} (total: {metadata['analyzed_subset_size']})\n")
        f.write(f"Translation Model: {metadata['translation_model']}\n")
        f.write(f"Embedding Model:  {metadata['embedding_model']}\n\n")
        
        # Performance metrics
        f.write("🏆 TRANSLATION QUALITY METRICS\n")
        f.write("-" * 50 + "\n")
        f.write("┌─────────────────────────┬─────────────┬─────────────┬─────────────┐\n")
        f.write("│ Metric                  │    Mean     │   Median    │    Std      │\n")
        f.write("├─────────────────────────┼─────────────┼─────────────┼─────────────┤\n")
        
        main_metrics = [
            ('Combined Score', 'combined_score'),
            ('Semantic Similarity', 'semantic_similarity'),
            ('ROUGE-1 F1', 'rouge1_f'),
            ('ROUGE-L F1', 'rougeL_f'),
            ('BERT F1', 'bert_f1'),
            ('BLEU Score', 'bleu_score'),
            ('Edit Distance', 'edit_distance_ratio'),
            ('Word F1', 'word_f1'),
        ]
        
        for name, key in main_metrics:
            mean_key = f'{key}_mean'
            median_key = f'{key}_median'
            std_key = f'{key}_std'
            
            if mean_key in stats and median_key in stats and std_key in stats:
                mean_val = stats[mean_key]
                median_val = stats[median_key]
                std_val = stats[std_key]
                f.write(f"│ {name:<23} │   {mean_val:7.3f}   │   {median_val:7.3f}   │   {std_val:7.3f}   │\n")
        
        f.write("└─────────────────────────┴─────────────┴─────────────┴─────────────┘\n\n")
        
        # Overall assessment
        if stats['valid_samples'] > 0:
            combined_mean = stats.get('combined_score_mean', 0)
            if combined_mean >= 0.85:
                quality = "🟢 EXCELLENT"
            elif combined_mean >= 0.75:
                quality = "🟡 GOOD"
            elif combined_mean >= 0.65:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 POOR"
            
            f.write(f"🏆 Overall Translation Quality: {quality} ({combined_mean:.1%})\n\n")
        
        # Sample breakdown (first 3 samples)
        valid_results = [r for r in results if r.get('error') is None]
        if valid_results:
            f.write("📋 SAMPLE BREAKDOWN (First 3 samples)\n")
            f.write("="*80 + "\n")
            
            for i, result in enumerate(valid_results[:3]):
                f.write(f"\n🔍 SAMPLE {i}:\n")
                f.write(f"❓ Original (EN): {result['original_english'][:80]}{'...' if len(result['original_english']) > 80 else ''}\n")
                f.write(f"🔄 Back-translated: {result['back_translation'][:80]}{'...' if len(result['back_translation']) > 80 else ''}\n")
                f.write(f"📊 Scores: Combined={result['combined_score']:.3f}, Semantic={result['metrics'].get('semantic_similarity', 0):.3f}, ROUGE-L={result['metrics'].get('rougeL_f', 0):.3f}\n")
                f.write("-" * 40 + "\n")
        
        f.write("\n" + "="*60 + "\n")

def print_summary(stats):
    """In tóm tắt kết quả với multiple metrics"""
    print("\n" + "="*80)
    print("BACK-TRANSLATION ANALYSIS - TÓM TẮT KẾT QUẢ")
    print("="*80)
    
    print(f"Tổng số samples: {stats['total_samples']}")
    print(f"Samples hợp lệ: {stats['valid_samples']}")
    print(f"Samples lỗi: {stats['error_samples']} ({stats['error_rate']:.1%})")
    
    if stats['valid_samples'] > 0:
        # Main metrics summary
        main_metrics = [
            ('Combined Score', 'combined_score'),
            ('Semantic Similarity', 'semantic_similarity'),
            ('ROUGE-1 F1', 'rouge1_f'),
            ('ROUGE-L F1', 'rougeL_f'),
            ('BERT F1', 'bert_f1'),
            ('Word F1', 'word_f1'),
            ('Edit Distance Ratio', 'edit_distance_ratio'),
            ('SacreBLEU', 'sacrebleu')
        ]
        
        print(f"\nCác chỉ số chính:")
        for name, key in main_metrics:
            if f'{key}_mean' in stats:
                print(f"  {name}: {stats[f'{key}_mean']:.4f} "
                      f"(min: {stats[f'{key}_min']:.4f}, max: {stats[f'{key}_max']:.4f})")
        
        print(f"\nPhân phối Consistency Level:")
        for level, data in stats['consistency_distribution'].items():
            print(f"  {level}: {data['count']} samples ({data['percentage']:.1f}%)")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Back-Translation Analysis với multiple metrics"
    )
    parser.add_argument('--data-path', type=str, default='data/translated_data.json')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='results/back_translation_analysis')
    parser.add_argument('--delay', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Kiểm tra API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Cần set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Load dữ liệu
    print("Đang load dữ liệu...")
    data = load_data(args.data_path)
    if data is None:
        sys.exit(1)
    
    # Xác định range
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(data)
    
    if start_idx < 0 or start_idx >= len(data):
        print(f"Lỗi: start index {start_idx} không hợp lệ")
        sys.exit(1)
    
    if end_idx <= start_idx or end_idx > len(data):
        print(f"Lỗi: end index {end_idx} không hợp lệ")
        sys.exit(1)
    
    subset_data = data[start_idx:end_idx]
    print(f"Sẽ phân tích {len(subset_data)} samples với multiple metrics...")
    
    # Khởi tạo analyzer
    analyzer = BackTranslationAnalyzer(api_key=api_key)
    
    # Phân tích từng sample
    print("Bắt đầu back-translation analysis...")
    results = []
    
    for i, sample in enumerate(subset_data):
        actual_index = start_idx + i
        print(f"Đang xử lý sample {actual_index} ({i+1}/{len(subset_data)})...", end=' ')
        
        result = analyzer.analyze_sample(sample, actual_index)
        results.append(result)
        
        if result['error']:
            print(f"LỖI: {result['error']}")
        else:
            metrics = result['metrics']
            print(f"Combined: {result['combined_score']:.3f}, "
                  f"ROUGE-1: {metrics['rouge1_f']:.3f}, "
                  f"BERT: {metrics['bert_f1']:.3f}, "
                  f"Level: {result['consistency_level']}")
        
        if i < len(subset_data) - 1:
            time.sleep(args.delay)
    
    # Tính thống kê
    print("\nĐang tính thống kê...")
    stats = calculate_statistics(results)
    
    # Tạo metadata
    timestamp = datetime.now().isoformat()
    metadata = {
        'timestamp': timestamp,
        'data_path': args.data_path,
        'start_index': start_idx,
        'end_index': end_idx,
        'total_dataset_size': len(data),
        'analyzed_subset_size': len(subset_data),
        'translation_model': analyzer.translation_model,
        'embedding_model': analyzer.embedding_model,
        'metrics_used': [
            'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
            'BERTScore', 'Edit Distance Ratio', 'SacreBLEU',
            'Jaccard Similarity', 'Word F1', 'Semantic Similarity'
        ],
        'parameters': {'delay': args.delay}
    }
    
    # Lưu kết quả
    folder_path = Path(args.output_dir) / f"back_translation_analysis_{start_idx}_{end_idx}"
    save_results(results, stats, folder_path, metadata)
    
    # In tóm tắt
    print_summary(stats)

if __name__ == "__main__":
    main()
