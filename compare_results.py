#!/usr/bin/env python3
"""
Compare results between different zero-shot evaluation methods
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from mint.evaluator import CypherEvaluator


def load_sample_results(folder_path: str) -> List[Dict[str, Any]]:
    """Load sample results from a folder"""
    results_path = Path("results") / folder_path / "sample_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Sample results file not found: {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_overview(folder_path: str) -> Dict[str, Any]:
    """Load overview information from a folder"""
    overview_path = Path("results") / folder_path / "overview.json"

    if not overview_path.exists():
        raise FileNotFoundError(f"Overview file not found: {overview_path}")

    with open(overview_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def align_samples(results1: List[Dict], results2: List[Dict]) -> tuple:
    """Align samples based on sample_id and return matched pairs"""
    # Create lookup dictionary for results2
    results2_dict = {sample['sample_id']: sample for sample in results2}

    aligned_results1 = []
    aligned_results2 = []

    for sample1 in results1:
        sample_id = sample1['sample_id']
        if sample_id in results2_dict:
            aligned_results1.append(sample1)
            aligned_results2.append(results2_dict[sample_id])

    return aligned_results1, aligned_results2


def compare_predictions(results1: List[Dict], results2: List[Dict],
                       method1_name: str, method2_name: str) -> Dict[str, Any]:
    """Compare predicted_cypher between two result sets"""

    # Align samples
    aligned1, aligned2 = align_samples(results1, results2)

    if not aligned1 or not aligned2:
        raise ValueError("No matching samples found between the two result sets")

    print(f"Comparing {len(aligned1)} matched samples")
    print(f"Method 1 ({method1_name}): {len(results1)} total samples")
    print(f"Method 2 ({method2_name}): {len(results2)} total samples")
    print(f"Matched samples: {len(aligned1)}")

    # Extract predicted_cypher from both methods
    predictions1 = [sample['predicted_cypher'] for sample in aligned1]
    predictions2 = [sample['predicted_cypher'] for sample in aligned2]

    # Use evaluator to compare predictions
    evaluator = CypherEvaluator()

    # Calculate metrics comparing method1 vs method2 predictions
    comparison_metrics = evaluator.evaluate_batch(predictions1, predictions2)

    # Calculate individual sample comparisons
    sample_comparisons = []
    for i, (sample1, sample2) in enumerate(zip(aligned1, aligned2)):
        pred1 = sample1['predicted_cypher']
        pred2 = sample2['predicted_cypher']

        # Individual comparison metrics
        individual_metrics = evaluator.evaluate_single(pred1, pred2)

        sample_comparison = {
            'sample_id': sample1['sample_id'],
            'question_en': sample1.get('question_en', ''),
            'question_vi': sample1.get('question_vi', ''),
            'gold_cypher': sample1.get('cypher', ''),
            f'{method1_name}_prediction': pred1,
            f'{method2_name}_prediction': pred2,
            'comparison_metrics': individual_metrics,
            f'{method1_name}_vs_gold': {
                'exact_match': sample1.get('exact_match', False),
                'bleu_score': sample1.get('bleu_score', 0.0),
                'syntax_valid': sample1.get('syntax_valid', False)
            },
            f'{method2_name}_vs_gold': {
                'exact_match': sample2.get('exact_match', False),
                'bleu_score': sample2.get('bleu_score', 0.0),
                'syntax_valid': sample2.get('syntax_valid', False)
            }
        }
        sample_comparisons.append(sample_comparison)

    # Calculate agreement metrics
    exact_matches = sum(1 for comp in sample_comparisons
                       if comp['comparison_metrics']['exact_match'])
    syntax_both_valid = sum(1 for comp in sample_comparisons
                           if comp[f'{method1_name}_vs_gold']['syntax_valid'] and
                              comp[f'{method2_name}_vs_gold']['syntax_valid'])

    both_correct = sum(1 for comp in sample_comparisons
                      if comp[f'{method1_name}_vs_gold']['exact_match'] and
                         comp[f'{method2_name}_vs_gold']['exact_match'])

    method1_better = sum(1 for comp in sample_comparisons
                        if comp[f'{method1_name}_vs_gold']['bleu_score'] >
                           comp[f'{method2_name}_vs_gold']['bleu_score'])

    method2_better = sum(1 for comp in sample_comparisons
                        if comp[f'{method2_name}_vs_gold']['bleu_score'] >
                           comp[f'{method1_name}_vs_gold']['bleu_score'])

    return {
        'comparison_overview': {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'total_matched_samples': len(aligned1),
            'prediction_agreement_rate': exact_matches / len(aligned1),
            'both_syntax_valid_rate': syntax_both_valid / len(aligned1),
            'both_correct_rate': both_correct / len(aligned1),
            'method1_better_count': method1_better,
            'method2_better_count': method2_better,
            'tie_count': len(aligned1) - method1_better - method2_better
        },
        'aggregate_comparison_metrics': comparison_metrics,
        'sample_comparisons': sample_comparisons
    }


def create_comparison_report(comparison_results: Dict[str, Any],
                           overview1: Dict, overview2: Dict,
                           output_folder: str):
    """Create and save comprehensive comparison report"""

    method1_name = comparison_results['comparison_overview']['method1_name']
    method2_name = comparison_results['comparison_overview']['method2_name']

    # Create summary report
    summary_report = {
        'comparison_info': {
            'timestamp': overview1['experiment_info']['timestamp'],
            'method1': {
                'name': method1_name,
                'model': overview1['experiment_info']['model'],
                'language': overview1['experiment_info']['language'],
                'total_samples': overview1['metrics']['total_samples']
            },
            'method2': {
                'name': method2_name,
                'model': overview2['experiment_info']['model'],
                'language': overview2['experiment_info']['language'],
                'total_samples': overview2['metrics']['total_samples']
            }
        },
        'comparison_summary': comparison_results['comparison_overview'],
        'prediction_similarity_metrics': {
            'exact_match_rate': comparison_results['aggregate_comparison_metrics']['exact_match_rate'],
            'average_bleu': comparison_results['aggregate_comparison_metrics']['average_bleu'],
            'average_rouge1': comparison_results['aggregate_comparison_metrics']['average_rouge1'],
            'average_rouge2': comparison_results['aggregate_comparison_metrics']['average_rouge2'],
            'average_rougeL': comparison_results['aggregate_comparison_metrics']['average_rougeL'],
            'average_meteor': comparison_results['aggregate_comparison_metrics']['average_meteor'],
            'average_bert_f1': comparison_results['aggregate_comparison_metrics']['average_bert_f1']
        },
        'performance_comparison': {
            f'{method1_name}_performance': {
                'exact_match_rate': overview1['metrics']['exact_match_rate'],
                'average_bleu': overview1['metrics']['average_bleu_score'],
                'syntax_valid_rate': overview1['metrics']['syntax_valid_rate']
            },
            f'{method2_name}_performance': {
                'exact_match_rate': overview2['metrics']['exact_match_rate'],
                'average_bleu': overview2['metrics']['average_bleu_score'],
                'syntax_valid_rate': overview2['metrics']['syntax_valid_rate']
            }
        }
    }

    # Save summary report
    summary_path = Path(output_folder) / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    # Save detailed comparison
    detailed_path = Path(output_folder) / "detailed_comparison.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    return summary_path, detailed_path


def print_comparison_summary(comparison_results: Dict[str, Any],
                           overview1: Dict, overview2: Dict):
    """Print comparison summary to console"""

    overview = comparison_results['comparison_overview']
    metrics = comparison_results['aggregate_comparison_metrics']

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY: {overview['method1_name']} vs {overview['method2_name']}")
    print(f"{'='*60}")

    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total matched samples: {overview['total_matched_samples']}")
    print(f"Prediction agreement rate: {overview['prediction_agreement_rate']:.4f}")
    print(f"Both methods syntax valid: {overview['both_syntax_valid_rate']:.4f}")
    print(f"Both methods correct: {overview['both_correct_rate']:.4f}")

    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"{overview['method1_name']} better: {overview['method1_better_count']} samples")
    print(f"{overview['method2_name']} better: {overview['method2_better_count']} samples")
    print(f"Ties: {overview['tie_count']} samples")

    print(f"\n📈 PREDICTION SIMILARITY METRICS:")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f}")
    print(f"BLEU Score: {metrics['average_bleu']:.4f}")
    print(f"ROUGE-1: {metrics['average_rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['average_rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['average_rougeL']:.4f}")
    print(f"METEOR: {metrics['average_meteor']:.4f}")
    print(f"BERTScore F1: {metrics['average_bert_f1']:.4f}")

    print(f"\n🔍 INDIVIDUAL PERFORMANCE:")
    print(f"{overview['method1_name']}:")
    print(f"  - Exact Match: {overview1['metrics']['exact_match_rate']:.4f}")
    print(f"  - BLEU Score: {overview1['metrics']['average_bleu_score']:.4f}")
    print(f"  - Syntax Valid: {overview1['metrics']['syntax_valid_rate']:.4f}")

    print(f"{overview['method2_name']}:")
    print(f"  - Exact Match: {overview2['metrics']['exact_match_rate']:.4f}")
    print(f"  - BLEU Score: {overview2['metrics']['average_bleu_score']:.4f}")
    print(f"  - Syntax Valid: {overview2['metrics']['syntax_valid_rate']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare results between different zero-shot evaluation methods")
    parser.add_argument("folder1", help="First result folder name (e.g., zero_shot_en_20250808_160330_0_500)")
    parser.add_argument("folder2", help="Second result folder name (e.g., zero_shot_vi_20250808_153911_0_500)")
    parser.add_argument("--output", default="comparison_results", help="Output folder name (default: comparison_results)")

    args = parser.parse_args()

    try:
        # Load results and overviews
        print("Loading results...")
        results1 = load_sample_results(args.folder1)
        results2 = load_sample_results(args.folder2)
        overview1 = load_overview(args.folder1)
        overview2 = load_overview(args.folder2)

        # Extract method names from folder names or overview
        method1_name = overview1['experiment_info'].get('language', 'method1')
        method2_name = overview2['experiment_info'].get('language', 'method2')

        # Perform comparison
        print("Performing comparison...")
        comparison_results = compare_predictions(results1, results2, method1_name, method2_name)

        # Create output folder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"results/{args.output}_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)

        # Create and save reports
        summary_path, detailed_path = create_comparison_report(
            comparison_results, overview1, overview2, output_folder
        )

        # Print summary
        print_comparison_summary(comparison_results, overview1, overview2)

        print(f"\n💾 RESULTS SAVED:")
        print(f"Summary report: {summary_path}")
        print(f"Detailed comparison: {detailed_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
