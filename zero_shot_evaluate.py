#!/usr/bin/env python3
"""
Zero-shot multilingual to Cypher conversion with enhanced evaluation metrics
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# Import enhanced evaluation modules
from mint.evaluator import CypherEvaluator
from mint.cypher_normalizer import CypherNormalizer

# Load environment variables
load_dotenv()

def generate_cypher(client: openai.OpenAI, question: str, schema: str, model: str, max_tokens: int) -> str:
    """Generate Cypher query from question using zero-shot prompting"""

    # Load template
    template_path = Path("templates/zero_shot_prompt.txt")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["schema", "question_vi"],
        template=template_content
    )

    prompt = prompt_template.format(
        schema=schema,
        question_vi=question
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0  # Use deterministic generation for consistency
        )

        generated_cypher = response.choices[0].message.content.strip()

        # Clean up the response - sometimes GPT adds extra text
        if "CYPHER QUERY:" in generated_cypher:
            generated_cypher = generated_cypher.split("CYPHER QUERY:")[-1].strip()

        # Remove code block markers if present
        generated_cypher = re.sub(r'^```(?:cypher)?\n?', '', generated_cypher)
        generated_cypher = re.sub(r'\n?```$', '', generated_cypher)

        return generated_cypher.strip()

    except Exception as e:
        print(f"Error generating Cypher: {e}")
        return ""

def create_results_folder(start: int, end: int, lang: str = 'vi') -> str:
    """Create timestamped results folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"zero_shot_{lang}_{timestamp}_{start}_{end}"
    results_path = Path("results") / folder_name
    results_path.mkdir(parents=True, exist_ok=True)
    return str(results_path)

def evaluate_zero_shot(start: int = 0, end: int = None, lang: str = 'vi'):
    """Evaluate zero-shot performance with enhanced metrics"""

    # Setup
    start_time = time.time()

    # Initialize enhanced evaluator
    evaluator = CypherEvaluator()

    # Load environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load Vietnamese data (should have both question_en and question_vi)
    data_path = "dataset/vietnamese_data.json"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Vietnamese data file not found: {data_path}")

    print(f"Using data source: {data_path}")
    print(f"Language: {'Vietnamese' if lang == 'vi' else 'English'}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply slice if specified
    if end is not None:
        data = data[start:end]
        print(f"Using data slice from {start} to {end-1}")
    elif start > 0:
        data = data[start:]
        print(f"Using data from index {start} to end")

    print(f"Evaluating zero-shot performance on {len(data)} samples")
    print(f"Using model: {model}")

    # Create results folder
    results_folder = create_results_folder(start, end if end else start + len(data), lang)
    print(f"Results will be saved to: {results_folder}")

    sample_results = []
    questions_used = 0
    predictions = []
    gold_standards = []

    for i, sample in enumerate(tqdm(data, desc="Generating Cypher queries")):
        # Get both question types from the sample
        question_vi = sample.get('question_vi', '')
        question_en = sample.get('question_en', '') or sample.get('question', '')

        # Choose question based on language parameter
        if lang == 'vi':
            if question_vi:
                question = question_vi
                questions_used += 1
            elif question_en:
                question = question_en
                if i < 5:  # Only show warning for first few samples
                    print(f"Warning: Using English question for sample {i} (no Vietnamese translation available)")
            else:
                print(f"Warning: No question found for sample {i}")
                continue
        else:  # lang == 'en'
            if question_en:
                question = question_en
                questions_used += 1
            elif question_vi:
                question = question_vi
                if i < 5:  # Only show warning for first few samples
                    print(f"Warning: Using Vietnamese question for sample {i} (no English translation available)")
            else:
                print(f"Warning: No question found for sample {i}")
                continue

        gold_cypher = sample.get('cypher', '')
        schema = sample.get('schema', '')

        # Generate Cypher using the selected question
        predicted_cypher = generate_cypher(client, question, schema, model, max_tokens)

        # Use enhanced evaluator for comprehensive metrics
        evaluation_result = evaluator.evaluate_single(predicted_cypher, gold_cypher)

        # Store sample result with all metrics
        sample_result = {
            'sample_id': start + i,
            'question_en': question_en,
            'question_vi': question_vi,
            'language': lang,
            'schema': schema,
            'cypher': gold_cypher,
            'predicted_cypher': predicted_cypher,
            **evaluation_result  # Include all evaluation metrics
        }
        sample_results.append(sample_result)

        # Collect for batch evaluation
        predictions.append(predicted_cypher)
        gold_standards.append(gold_cypher)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate batch metrics (including BERTScore)
    batch_metrics = evaluator.evaluate_batch(predictions, gold_standards)

    # Calculate overall metrics
    total_samples = len(sample_results)
    language_coverage = questions_used / total_samples if total_samples > 0 else 0

    print(f"\nENHANCED EVALUATION RESULTS:")
    print(f"Total samples: {total_samples}")
    print(f"Language: {'Vietnamese' if lang == 'vi' else 'English'}")
    print(f"{lang.upper()} questions used: {questions_used} ({language_coverage:.2%})")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"\n=== TEXT-BASED METRICS ===")
    print(f"Exact Match (Normalized): {batch_metrics['exact_match_rate']:.4f}")
    print(f"Exact Match (Raw): {batch_metrics['exact_match_raw_rate']:.4f}")
    print(f"BLEU Score: {batch_metrics['average_bleu']:.4f}")
    print(f"ROUGE-1: {batch_metrics['average_rouge1']:.4f}")
    print(f"ROUGE-2: {batch_metrics['average_rouge2']:.4f}")
    print(f"ROUGE-L: {batch_metrics['average_rougeL']:.4f}")
    print(f"METEOR Score: {batch_metrics['average_meteor']:.4f}")
    print(f"\n=== EMBEDDING-BASED METRICS ===")
    print(f"BERTScore Precision: {batch_metrics['average_bert_precision']:.4f}")
    print(f"BERTScore Recall: {batch_metrics['average_bert_recall']:.4f}")
    print(f"BERTScore F1: {batch_metrics['average_bert_f1']:.4f}")
    print(f"\n=== SEMANTIC METRICS ===")
    print(f"Semantic Similarity: {batch_metrics['average_semantic_similarity']:.4f}")
    print(f"\n=== SYNTAX VALIDATION ===")
    print(f"Syntax Valid Rate: {batch_metrics['syntax_valid_rate']:.4f}")

    # Save overview file with enhanced metrics
    overview_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'max_tokens': max_tokens,
            'language': lang,
            'start_index': start,
            'end_index': start + len(data),
            'data_source': 'vietnamese_data.json'
        },
        'metrics': {
            'total_samples': total_samples,
            'language': lang,
            'questions_used': questions_used,
            'language_coverage': language_coverage,
            'execution_time_seconds': execution_time,

            # Text-based metrics
            'exact_match_rate': batch_metrics['exact_match_rate'],
            'exact_match_raw_rate': batch_metrics['exact_match_raw_rate'],
            'average_bleu_score': batch_metrics['average_bleu'],
            'average_rouge1': batch_metrics['average_rouge1'],
            'average_rouge2': batch_metrics['average_rouge2'],
            'average_rougeL': batch_metrics['average_rougeL'],
            'average_meteor': batch_metrics['average_meteor'],

            # Embedding-based metrics
            'average_bert_precision': batch_metrics['average_bert_precision'],
            'average_bert_recall': batch_metrics['average_bert_recall'],
            'average_bert_f1': batch_metrics['average_bert_f1'],

            # Semantic metrics
            'average_semantic_similarity': batch_metrics['average_semantic_similarity'],

            # Syntax validation
            'syntax_valid_rate': batch_metrics['syntax_valid_rate']
        }
    }

    overview_path = Path(results_folder) / "overview.json"
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(overview_data, f, ensure_ascii=False, indent=2)

    # Save detailed sample results
    samples_path = Path(results_folder) / "sample_results.json"
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(sample_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {results_folder}")
    print(f"Overview: {overview_path}")
    print(f"Detailed results: {samples_path}")

def main():
    parser = argparse.ArgumentParser(description="Zero-shot multilingual to Cypher evaluation with enhanced metrics")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, help="End index (default: all samples)")
    parser.add_argument("--lang", choices=['vi', 'en'], default='vi',
                       help="Language to use for questions: vi (Vietnamese) or en (English). Default: vi")

    args = parser.parse_args()

    evaluate_zero_shot(args.start, args.end, args.lang)

if __name__ == "__main__":
    main()
