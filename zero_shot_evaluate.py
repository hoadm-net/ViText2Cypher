#!/usr/bin/env python3
"""
Zero-shot multilingual to Cypher conversion with evaluation
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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load environment variables
load_dotenv()

def setup_nltk():
    """Setup NLTK requirements"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def normalize_cypher(cypher: str) -> str:
    """Normalize Cypher query for comparison"""
    if not cypher:
        return ""

    # Remove extra whitespace and newlines
    normalized = re.sub(r'\s+', ' ', cypher.strip())

    # Remove trailing semicolon if present
    normalized = normalized.rstrip(';')

    # Convert to lowercase for comparison
    normalized = normalized.lower()

    return normalized

def calculate_exact_match(predicted: str, gold: str) -> bool:
    """Calculate exact match between predicted and gold Cypher"""
    pred_norm = normalize_cypher(predicted)
    gold_norm = normalize_cypher(gold)
    return pred_norm == gold_norm

def calculate_bleu(predicted: str, gold: str) -> float:
    """Calculate BLEU score between predicted and gold Cypher"""
    if not predicted or not gold:
        return 0.0

    # Tokenize the queries
    pred_tokens = predicted.split()
    gold_tokens = [gold.split()]  # BLEU expects list of reference sentences

    # Use smoothing function to handle edge cases
    smoothing = SmoothingFunction().method1

    try:
        score = sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

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
    """Evaluate zero-shot performance on questions"""

    # Setup
    setup_nltk()
    start_time = time.time()

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
    exact_matches = 0
    total_bleu = 0.0
    questions_used = 0

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

        # Calculate metrics
        exact_match = calculate_exact_match(predicted_cypher, gold_cypher)
        bleu_score = calculate_bleu(predicted_cypher, gold_cypher)

        if exact_match:
            exact_matches += 1
        total_bleu += bleu_score

        # Store sample result
        sample_result = {
            'sample_id': start + i,
            'question_en': question_en,
            'question_vi': question_vi,
            'language': lang,
            'schema': schema,
            'cypher': gold_cypher,
            'predicted_cypher': predicted_cypher,
            'exact_match': exact_match,
            'bleu_score': bleu_score
        }
        sample_results.append(sample_result)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate overall metrics
    total_samples = len(sample_results)
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0
    language_coverage = questions_used / total_samples if total_samples > 0 else 0

    print(f"\nEVALUATION RESULTS:")
    print(f"Total samples: {total_samples}")
    print(f"Language: {'Vietnamese' if lang == 'vi' else 'English'}")
    print(f"{lang.upper()} questions used: {questions_used} ({language_coverage:.2%})")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Exact Match: {exact_matches}/{total_samples} ({exact_match_rate:.4f})")
    print(f"Average BLEU: {avg_bleu:.4f}")

    # Save overview file
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
            'exact_match_count': exact_matches,
            'exact_match_rate': exact_match_rate,
            'average_bleu_score': avg_bleu
        }
    }

    overview_path = Path(results_folder) / "overview.json"
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(overview_data, f, ensure_ascii=False, indent=2)

    # Save detailed sample results
    samples_path = Path(results_folder) / "sample_results.json"
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(sample_results, f, ensure_ascii=False, indent=2)

    print(f"Overview saved to: {overview_path}")
    print(f"Sample results saved to: {samples_path}")

def main():
    parser = argparse.ArgumentParser(description="Zero-shot multilingual to Cypher evaluation")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, help="End index (default: all samples)")
    parser.add_argument("--lang", choices=['vi', 'en'], default='vi',
                       help="Language to use for questions: vi (Vietnamese) or en (English). Default: vi")

    args = parser.parse_args()

    evaluate_zero_shot(args.start, args.end, args.lang)

if __name__ == "__main__":
    main()
