"""
Cypher evaluation utilities for measuring LLM-generated query quality
"""

import re
from typing import Set, Any, Union, Dict, List, Tuple, Hashable
import textdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_jw_distance(string1: str, string2: str) -> float:
    """
    Calculate the Jaro-Winkler distance between two strings.

    The Jaro-Winkler distance is a measure of similarity between two strings.
    The score is normalized such that 0 equates to no similarity and
    1 is an exact match.
    """
    return textdistance.jaro_winkler(string1, string2)


def normalize_cypher(cypher: str) -> str:
    """
    Normalize Cypher query for comparison by:
    - Removing extra whitespace and newlines
    - Converting to lowercase
    - Removing trailing semicolons
    """
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
    """Calculate exact match between predicted and gold Cypher after normalization"""
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


def rowsim(setL: Set, setR: Set) -> float:
    """Calculate the similarity between two sets using Jaccard index formula"""
    if not setL and not setR:
        return 1.0
    if not setL or not setR:
        return 0.0
    return len(setL.intersection(setR)) / len(setL.union(setR))


def floatify(v: Any) -> Any:
    """
    Attempts to convert a value to a float if it represents a number,
    or recursively apply the conversion to elements within a list or dict.
    """
    if isinstance(v, str):
        return v
    try:
        f = float(v)
        return f
    except:
        pass
    if isinstance(v, list):
        return [floatify(x) for x in v]
    if isinstance(v, dict):
        return {k: floatify(u) for k, u in v.items()}
    return v


def make_hashable(v: Any) -> Hashable:
    """Convert a value to a hashable type (needed for set operations)"""
    float_v = floatify(v)
    if not isinstance(float_v, Hashable):
        return str(float_v)
    else:
        return float_v


def make_alignment(dictL: List[Dict], dictR: List[Dict]) -> Tuple[List[Set], List[Set]]:
    """Align rows from two lists of dictionaries based on their similarity"""
    swap = len(dictL) > len(dictR)

    # Forming set views from the list of dictionaries
    setViewsL = [{make_hashable(v) for k, v in row.items()} for row in dictL]
    setViewsR = [{make_hashable(v) for k, v in row.items()} for row in dictR]
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL

    for i in range(len(setViewsL)):
        max_sim = -1
        max_j = -1
        for j in range(i, len(setViewsR)):
            sim = rowsim(setViewsL[i], setViewsR[j])
            if sim > max_sim:
                max_j = j
                max_sim = sim
        tmp = setViewsR[i]
        setViewsR[i] = setViewsR[max_j]
        setViewsR[max_j] = tmp
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL
    return setViewsL, setViewsR


def df_sim(dictL: List[Dict], dictR: List[Dict], list_view: bool) -> float:
    """
    Calculate the data frame similarity based on either the original row order or an alignment.
    """
    if list_view:
        # Original row order for lists of dictionaries
        view_L = [row.values() for row in dictL]
        view_R = [row.values() for row in dictR]
    else:
        view_L, view_R = make_alignment(dictL, dictR)

    totalSetL = set()
    for i, s in enumerate(view_L):
        for elem in s:
            totalSetL.add((i, make_hashable(elem)))
    totalSetR = set()
    for i, s in enumerate(view_R):
        for elem in s:
            totalSetR.add((i, make_hashable(elem)))
    intersection = totalSetL.intersection(totalSetR)
    union = totalSetL.union(totalSetR)

    if len(union) == 0 and len(intersection) == 0:
        return 1.0
    elif len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def df_sim_pair(pair_L, pair_R):
    """
    Compute the Jaccard similarity of two data frames (lists of dictionaries),
    taking into account the order of rows if indicated by the involved Cypher queries.
    """
    cypher_L, dict_L = pair_L
    cypher_R, dict_R = pair_R

    return df_sim(dict_L, dict_R, "order by" in f"{cypher_L} {cypher_R}".lower())


def evaluate_cypher_queries(predicted_queries: List[str],
                          gold_queries: List[str],
                          predicted_results: List[List[Dict]],
                          gold_results: List[List[Dict]]) -> Dict[str, float]:
    """
    Evaluate a batch of Cypher queries using multiple metrics

    Args:
        predicted_queries: List of predicted Cypher queries
        gold_queries: List of gold standard Cypher queries
        predicted_results: List of result sets from predicted queries
        gold_results: List of result sets from gold queries

    Returns:
        Dictionary containing evaluation metrics
    """
    if len(predicted_queries) != len(gold_queries):
        raise ValueError("Number of predicted and gold queries must match")

    jw_scores = []
    exact_matches = []
    bleu_scores = []
    jaccard_scores = []

    for i in range(len(predicted_queries)):
        # Jaro-Winkler similarity
        jw_score = get_jw_distance(predicted_queries[i], gold_queries[i])
        jw_scores.append(jw_score)

        # Exact match
        exact_match = calculate_exact_match(predicted_queries[i], gold_queries[i])
        exact_matches.append(1.0 if exact_match else 0.0)

        # BLEU score
        bleu_score = calculate_bleu(predicted_queries[i], gold_queries[i])
        bleu_scores.append(bleu_score)

        # Jaccard similarity on results
        if i < len(predicted_results) and i < len(gold_results):
            jaccard_score = df_sim_pair(
                (predicted_queries[i], predicted_results[i]),
                (gold_queries[i], gold_results[i])
            )
            jaccard_scores.append(jaccard_score)
        else:
            jaccard_scores.append(0.0)

    return {
        'avg_jaro_winkler': sum(jw_scores) / len(jw_scores) if jw_scores else 0.0,
        'exact_match_rate': sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        'avg_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        'avg_jaccard': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
        'total_samples': len(predicted_queries)
    }
