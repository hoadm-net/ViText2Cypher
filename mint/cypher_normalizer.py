#!/usr/bin/env python3
"""
Cypher query normalization and comparison utilities
"""

import re
import string
from typing import List, Set, Tuple
from difflib import SequenceMatcher


class CypherNormalizer:
    """Advanced Cypher query normalization for fair comparison"""

    def __init__(self):
        self.cypher_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'UNWIND', 'CREATE', 'MERGE',
            'DELETE', 'SET', 'REMOVE', 'ORDER', 'BY', 'LIMIT', 'SKIP',
            'UNION', 'ALL', 'OPTIONAL', 'FOREACH', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'AND', 'OR', 'NOT', 'XOR', 'IN', 'STARTS',
            'ENDS', 'CONTAINS', 'IS', 'NULL', 'DISTINCT', 'AS', 'ASC', 'DESC'
        }

    def normalize_whitespace(self, query: str) -> str:
        """Normalize whitespace in Cypher query"""
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', query.strip())

        # Normalize around operators and punctuation
        normalized = re.sub(r'\s*([(){}[\],;])\s*', r'\1', normalized)
        normalized = re.sub(r'\s*([<>=!]+)\s*', r' \1 ', normalized)

        return normalized.strip()

    def normalize_case(self, query: str) -> str:
        """Normalize case for Cypher keywords while preserving property names"""
        tokens = query.split()
        normalized_tokens = []

        for token in tokens:
            # Check if token is a Cypher keyword
            clean_token = token.strip('(){}[],:;').upper()
            if clean_token in self.cypher_keywords:
                normalized_tokens.append(clean_token)
            else:
                # Preserve original case for non-keywords
                normalized_tokens.append(token)

        return ' '.join(normalized_tokens)

    def remove_optional_elements(self, query: str) -> str:
        """Remove optional elements that don't affect query semantics"""
        # Remove trailing semicolon
        query = query.rstrip(';')

        # Normalize quotes (single vs double)
        query = re.sub(r"'([^']*)'", r'"\1"', query)

        return query

    def extract_query_structure(self, query: str) -> str:
        """Extract the structural elements of a query for comparison"""
        # Remove string literals and replace with placeholder
        query = re.sub(r'"[^"]*"', '"STRING"', query)
        query = re.sub(r"'[^']*'", '"STRING"', query)

        # Remove numeric literals
        query = re.sub(r'\b\d+\.?\d*\b', 'NUMBER', query)

        return query

    def normalize_query(self, query: str, level: str = 'standard') -> str:
        """
        Normalize Cypher query with different levels of normalization

        Args:
            query: Input Cypher query
            level: 'basic', 'standard', or 'strict'
        """
        if not query:
            return ""

        normalized = query

        # Basic normalization
        normalized = self.normalize_whitespace(normalized)
        normalized = self.remove_optional_elements(normalized)

        if level in ['standard', 'strict']:
            normalized = self.normalize_case(normalized)

        if level == 'strict':
            normalized = self.extract_query_structure(normalized)

        return normalized.lower() if level == 'strict' else normalized


def calculate_semantic_similarity(query1: str, query2: str) -> float:
    """
    Calculate semantic similarity between two Cypher queries
    by comparing their normalized structures
    """
    normalizer = CypherNormalizer()

    # Normalize both queries
    norm1 = normalizer.normalize_query(query1, level='strict')
    norm2 = normalizer.normalize_query(query2, level='strict')

    # Use sequence matcher for similarity
    matcher = SequenceMatcher(None, norm1, norm2)
    return matcher.ratio()


def extract_cypher_components(query: str) -> dict:
    """
    Extract key components from a Cypher query for detailed analysis
    """
    components = {
        'match_clauses': [],
        'where_clauses': [],
        'return_clauses': [],
        'with_clauses': [],
        'order_by': [],
        'limit': None,
        'functions': [],
        'operators': []
    }

    # Extract MATCH clauses
    match_pattern = r'MATCH\s+([^WHERE^RETURN^WITH^ORDER^LIMIT]+)'
    matches = re.findall(match_pattern, query, re.IGNORECASE)
    components['match_clauses'] = [m.strip() for m in matches]

    # Extract WHERE clauses
    where_pattern = r'WHERE\s+([^RETURN^WITH^ORDER^LIMIT]+)'
    wheres = re.findall(where_pattern, query, re.IGNORECASE)
    components['where_clauses'] = [w.strip() for w in wheres]

    # Extract RETURN clauses
    return_pattern = r'RETURN\s+([^ORDER^LIMIT]+)'
    returns = re.findall(return_pattern, query, re.IGNORECASE)
    components['return_clauses'] = [r.strip() for r in returns]

    # Extract functions
    function_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX|COLLECT|DISTINCT)\s*\('
    functions = re.findall(function_pattern, query, re.IGNORECASE)
    components['functions'] = list(set(f.upper() for f in functions))

    # Extract LIMIT
    limit_pattern = r'LIMIT\s+(\d+)'
    limit_match = re.search(limit_pattern, query, re.IGNORECASE)
    if limit_match:
        components['limit'] = int(limit_match.group(1))

    return components


def compare_query_components(query1: str, query2: str) -> dict:
    """
    Compare two Cypher queries component by component
    """
    comp1 = extract_cypher_components(query1)
    comp2 = extract_cypher_components(query2)

    comparison = {}

    # Compare each component type
    for component_type in comp1.keys():
        if component_type == 'limit':
            comparison[f'{component_type}_match'] = comp1[component_type] == comp2[component_type]
        elif isinstance(comp1[component_type], list):
            set1 = set(comp1[component_type])
            set2 = set(comp2[component_type])
            if set1 or set2:
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
            else:
                jaccard = 1.0
            comparison[f'{component_type}_similarity'] = jaccard

    return comparison


def analyze_query_complexity(query: str) -> dict:
    """
    Analyze the complexity of a Cypher query
    """
    complexity = {
        'num_match_clauses': len(re.findall(r'\bMATCH\b', query, re.IGNORECASE)),
        'num_where_clauses': len(re.findall(r'\bWHERE\b', query, re.IGNORECASE)),
        'num_with_clauses': len(re.findall(r'\bWITH\b', query, re.IGNORECASE)),
        'has_aggregation': bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\b', query, re.IGNORECASE)),
        'has_ordering': bool(re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE)),
        'has_limit': bool(re.search(r'\bLIMIT\b', query, re.IGNORECASE)),
        'has_optional': bool(re.search(r'\bOPTIONAL\b', query, re.IGNORECASE)),
        'num_relationships': len(re.findall(r'-\[.*?\]-', query)),
        'query_length': len(query.split())
    }

    # Calculate overall complexity score
    complexity_score = (
        complexity['num_match_clauses'] * 1 +
        complexity['num_where_clauses'] * 1 +
        complexity['num_with_clauses'] * 2 +
        complexity['has_aggregation'] * 2 +
        complexity['has_ordering'] * 1 +
        complexity['has_optional'] * 2 +
        complexity['num_relationships'] * 1 +
        complexity['query_length'] * 0.1
    )

    complexity['complexity_score'] = complexity_score
    return complexity


if __name__ == "__main__":
    # Test the normalization
    test_queries = [
        "MATCH (n:Person) WHERE n.age > 25 RETURN n.name;",
        "match (n:Person) where n.age>25 return n.name",
        "MATCH (n:Person) WHERE n.age>25 RETURN n.name ORDER BY n.name",
    ]

    normalizer = CypherNormalizer()

    for query in test_queries:
        print(f"Original: {query}")
        print(f"Normalized: {normalizer.normalize_query(query)}")
        print(f"Strict: {normalizer.normalize_query(query, 'strict')}")
        print("---")
