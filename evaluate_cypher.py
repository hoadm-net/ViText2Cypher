#!/usr/bin/env python3
"""
Enhanced Cypher evaluation script for ViText2Cypher project
Supports both Vietnamese and English evaluation with comprehensive metrics
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# Import our evaluation utilities
from mint.evaluator import (
    get_jw_distance, calculate_exact_match, calculate_bleu,
    df_sim_pair, evaluate_cypher_queries, normalize_cypher
)


def load_evaluation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_cypher_response(response: str) -> str:
    """Clean up LLM-generated Cypher response"""
    if not response:
        return ""

    # Remove code block markers
    response = response.replace('```cypher', '').replace('```', '')

    # Remove common prefixes
    if "CYPHER QUERY:" in response:
        response = response.split("CYPHER QUERY:")[-1]

    return response.strip()


def evaluate_single_sample(sample: Dict[str, Any],
                         predicted_cypher: str,
                         execute_queries: bool = False,
                         graph_connection=None) -> Dict[str, Any]:
    """
    Evaluate a single sample with predicted Cypher

    Args:
        sample: Sample data containing question, schema, gold cypher
        predicted_cypher: Generated Cypher query
        execute_queries: Whether to execute queries against database
        graph_connection: Database connection if execute_queries=True
    """
    gold_cypher = sample.get('cypher', '')
    question_vi = sample.get('question_vi', '')
    question_en = sample.get('question', '')
    schema = sample.get('schema', '')

    # Clean the predicted cypher
    predicted_cypher = clean_cypher_response(predicted_cypher)

    # Calculate string-based metrics
    jw_score = get_jw_distance(predicted_cypher, gold_cypher)
    exact_match = calculate_exact_match(predicted_cypher, gold_cypher)
    bleu_score = calculate_bleu(predicted_cypher, gold_cypher)

    result = {
        'question_en': question_en,
        'question_vi': question_vi,
        'schema': schema,
        'gold_cypher': gold_cypher,
        'predicted_cypher': predicted_cypher,
        'jaro_winkler': jw_score,
        'exact_match': exact_match,
        'bleu_score': bleu_score,
        'jaccard_score': 0.0,  # Default if no execution
        'execution_success': False,
        'error_message': None
    }

    # Execute queries if requested and connection available
    if execute_queries and graph_connection:
        try:
            # Execute gold query
            gold_results = graph_connection.query(gold_cypher)
            result['gold_results'] = gold_results

            # Execute predicted query
            predicted_results = graph_connection.query(predicted_cypher)
            result['predicted_results'] = predicted_results
            result['execution_success'] = True

            # Calculate Jaccard similarity on results
            jaccard_score = df_sim_pair(
                (predicted_cypher, predicted_results),
                (gold_cypher, gold_results)
            )
            result['jaccard_score'] = jaccard_score

        except Exception as e:
            result['error_message'] = str(e)
            result['jaccard_score'] = 0.0

    return result


def create_evaluation_report(results: List[Dict[str, Any]],
                           output_dir: str,
                           experiment_name: str) -> Dict[str, float]:
    """Create comprehensive evaluation report"""

    # Calculate overall metrics
    total_samples = len(results)

    metrics = {
        'total_samples': total_samples,
        'avg_jaro_winkler': sum(r['jaro_winkler'] for r in results) / total_samples,
        'exact_match_rate': sum(r['exact_match'] for r in results) / total_samples,
        'avg_bleu': sum(r['bleu_score'] for r in results) / total_samples,
        'avg_jaccard': sum(r['jaccard_score'] for r in results) / total_samples,
        'execution_success_rate': sum(r['execution_success'] for r in results) / total_samples
    }

    # Create detailed DataFrame for analysis
    df = pd.DataFrame(results)

    # Save detailed results
    detailed_path = os.path.join(output_dir, f"{experiment_name}_detailed_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save summary metrics
    summary_path = os.path.join(output_dir, f"{experiment_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save CSV for easy analysis
    csv_path = os.path.join(output_dir, f"{experiment_name}_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')

    return metrics


def evaluate_translation_quality(data_path: str) -> Dict[str, float]:
    """
    Evaluate quality of Vietnamese translations
    Compare Vietnamese questions with English originals
    """
    data = load_evaluation_data(data_path)

    translation_scores = []

    for sample in data:
        question_en = sample.get('question', '')
        question_vi = sample.get('question_vi', '')

        if question_en and question_vi:
            # Simple length ratio check
            len_ratio = len(question_vi) / len(question_en) if len(question_en) > 0 else 0
            translation_scores.append({
                'length_ratio': len_ratio,
                'has_vietnamese': bool(question_vi),
                'has_english': bool(question_en)
            })

    if not translation_scores:
        return {'translation_coverage': 0.0, 'avg_length_ratio': 0.0}

    return {
        'translation_coverage': sum(s['has_vietnamese'] for s in translation_scores) / len(translation_scores),
        'avg_length_ratio': sum(s['length_ratio'] for s in translation_scores) / len(translation_scores),
        'total_samples': len(translation_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Cypher evaluation for ViText2Cypher")
    parser.add_argument("--data", required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--predictions", help="Path to predictions JSON file (if separate)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--experiment-name", help="Name for this experiment")
    parser.add_argument("--execute-queries", action="store_true", help="Execute queries against database")
    parser.add_argument("--check-translations", action="store_true", help="Check translation quality")
    parser.add_argument("--sample-size", type=int, help="Limit evaluation to N samples")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"cypher_eval_{timestamp}"

    print(f"Starting evaluation: {args.experiment_name}")

    # Load data
    data = load_evaluation_data(args.data)

    if args.sample_size:
        data = data[:args.sample_size]

    print(f"Evaluating {len(data)} samples")

    # Check translation quality if requested
    if args.check_translations:
        print("Checking translation quality...")
        translation_metrics = evaluate_translation_quality(args.data)
        print(f"Translation coverage: {translation_metrics['translation_coverage']:.2%}")
        print(f"Average length ratio: {translation_metrics['avg_length_ratio']:.2f}")

    # For demonstration, we'll simulate predicted queries
    # In real usage, these would come from your LLM
    results = []

    for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        # For demo purposes, use the gold cypher with some noise
        # In real usage, this would be your LLM-generated cypher
        predicted_cypher = sample.get('cypher', '')

        # Add some realistic variations for demonstration
        if i % 3 == 0:
            # Simulate minor formatting differences
            predicted_cypher = predicted_cypher.lower()
        elif i % 3 == 1:
            # Simulate missing semicolon
            predicted_cypher = predicted_cypher.rstrip(';')

        result = evaluate_single_sample(
            sample=sample,
            predicted_cypher=predicted_cypher,
            execute_queries=args.execute_queries
        )

        result['sample_id'] = i
        results.append(result)

    # Create evaluation report
    metrics = create_evaluation_report(results, args.output_dir, args.experiment_name)

    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Jaro-Winkler similarity: {metrics['avg_jaro_winkler']:.4f}")
    print(f"Exact match rate: {metrics['exact_match_rate']:.4f}")
    print(f"Average BLEU score: {metrics['avg_bleu']:.4f}")
    print(f"Average Jaccard score: {metrics['avg_jaccard']:.4f}")

    if args.execute_queries:
        print(f"Query execution success rate: {metrics['execution_success_rate']:.4f}")

    print(f"\nResults saved to: {args.output_dir}/{args.experiment_name}_*")


if __name__ == "__main__":
    main()
