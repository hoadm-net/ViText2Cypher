"""
Enhanced Cypher evaluation utilities with comprehensive metrics
"""

import re
from typing import Set, Any, Union, Dict, List, Tuple, Hashable, Optional
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from .cypher_normalizer import CypherNormalizer

# Import additional libraries for new metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class CypherEvaluator:
    """Enhanced Cypher evaluation with multiple metrics"""

    def __init__(self):
        self.normalizer = CypherNormalizer()
        self.setup_nltk()

        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def setup_nltk(self):
        """Setup NLTK requirements"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')

    def calculate_exact_match(self, predicted: str, gold: str, normalize: bool = True) -> bool:
        """Calculate exact match with optional normalization"""
        if normalize:
            pred_norm, gold_norm = self.normalizer.normalize_query_pair(predicted, gold)
            return pred_norm.lower() == gold_norm.lower()
        else:
            return predicted.strip() == gold.strip()

    def calculate_bleu(self, predicted: str, gold: str) -> float:
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

    def calculate_rouge(self, predicted: str, gold: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        if not predicted or not gold:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        try:
            scores = self.rouge_scorer.score(gold, predicted)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def calculate_meteor(self, predicted: str, gold: str) -> float:
        """Calculate METEOR score"""
        if not predicted or not gold:
            return 0.0

        try:
            # Tokenize
            pred_tokens = nltk.word_tokenize(predicted.lower())
            gold_tokens = nltk.word_tokenize(gold.lower())

            score = meteor_score([gold_tokens], pred_tokens)
            return score
        except:
            return 0.0

    def calculate_bert_score(self, predicted: List[str], gold: List[str]) -> Dict[str, List[float]]:
        """Calculate BERTScore for batch of predictions"""
        if not BERTSCORE_AVAILABLE:
            return {'precision': [0.0] * len(predicted), 'recall': [0.0] * len(predicted), 'f1': [0.0] * len(predicted)}

        if not predicted or not gold or len(predicted) != len(gold):
            return {'precision': [0.0] * len(predicted), 'recall': [0.0] * len(predicted), 'f1': [0.0] * len(predicted)}

        try:
            P, R, F1 = bert_score(predicted, gold, lang='en', verbose=False)
            return {
                'precision': P.tolist(),
                'recall': R.tolist(),
                'f1': F1.tolist()
            }
        except:
            return {'precision': [0.0] * len(predicted), 'recall': [0.0] * len(predicted), 'f1': [0.0] * len(predicted)}

    def validate_cypher_syntax(self, query: str) -> Tuple[bool, str]:
        """Validate Cypher syntax using normalizer"""
        return self.normalizer.validate_cypher_syntax(query)

    def evaluate_single(self, predicted: str, gold: str) -> Dict[str, Any]:
        """Comprehensive evaluation of a single prediction"""
        # Syntax validation
        is_valid, syntax_error = self.validate_cypher_syntax(predicted)

        # Text-based metrics
        exact_match = self.calculate_exact_match(predicted, gold)
        exact_match_raw = self.calculate_exact_match(predicted, gold, normalize=False)
        bleu_score = self.calculate_bleu(predicted, gold)
        rouge_scores = self.calculate_rouge(predicted, gold)
        meteor_score = self.calculate_meteor(predicted, gold)

        # Semantic similarity
        semantic_similarity = self.normalizer.calculate_semantic_similarity(predicted, gold)

        return {
            'exact_match': exact_match,
            'exact_match_raw': exact_match_raw,
            'bleu_score': bleu_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'meteor_score': meteor_score,
            'semantic_similarity': semantic_similarity,
            'syntax_valid': is_valid,
            'syntax_error': syntax_error if not is_valid else None
        }

    def evaluate_batch(self, predictions: List[str], gold_standards: List[str]) -> Dict[str, Any]:
        """Evaluate a batch of predictions with all metrics"""
        if len(predictions) != len(gold_standards):
            raise ValueError("Predictions and gold standards must have same length")

        # Individual metrics
        results = []
        for pred, gold in zip(predictions, gold_standards):
            results.append(self.evaluate_single(pred, gold))

        # Calculate BERTScore for entire batch (more efficient)
        bert_scores = self.calculate_bert_score(predictions, gold_standards)

        # Add BERTScore to individual results
        for i, result in enumerate(results):
            result['bert_precision'] = bert_scores['precision'][i]
            result['bert_recall'] = bert_scores['recall'][i]
            result['bert_f1'] = bert_scores['f1'][i]

        # Aggregate metrics
        total_samples = len(results)
        metrics = {
            'total_samples': total_samples,
            'exact_match_rate': sum(r['exact_match'] for r in results) / total_samples,
            'exact_match_raw_rate': sum(r['exact_match_raw'] for r in results) / total_samples,
            'average_bleu': sum(r['bleu_score'] for r in results) / total_samples,
            'average_rouge1': sum(r['rouge1'] for r in results) / total_samples,
            'average_rouge2': sum(r['rouge2'] for r in results) / total_samples,
            'average_rougeL': sum(r['rougeL'] for r in results) / total_samples,
            'average_meteor': sum(r['meteor_score'] for r in results) / total_samples,
            'average_semantic_similarity': sum(r['semantic_similarity'] for r in results) / total_samples,
            'syntax_valid_rate': sum(r['syntax_valid'] for r in results) / total_samples,
            'average_bert_precision': sum(bert_scores['precision']) / len(bert_scores['precision']),
            'average_bert_recall': sum(bert_scores['recall']) / len(bert_scores['recall']),
            'average_bert_f1': sum(bert_scores['f1']) / len(bert_scores['f1']),
            'individual_results': results
        }

        return metrics

    # Keep existing methods for backward compatibility
    def get_jw_distance(self, string1: str, string2: str) -> float:
        """Calculate the Jaro-Winkler distance between two strings."""
        try:
            import textdistance
            return textdistance.jaro_winkler(string1, string2)
        except ImportError:
            # Fallback to simple similarity
            return self.normalizer.calculate_semantic_similarity(string1, string2)

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
