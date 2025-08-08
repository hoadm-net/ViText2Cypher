#!/usr/bin/env python3
"""
Enhanced Cypher query normalization and comparison utilities
"""

import re
import string
from typing import List, Set, Tuple, Dict
from difflib import SequenceMatcher


class CypherNormalizer:
    """Advanced Cypher query normalization for fair comparison"""

    def __init__(self):
        self.cypher_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'UNWIND', 'CREATE', 'MERGE',
            'DELETE', 'SET', 'REMOVE', 'ORDER', 'BY', 'LIMIT', 'SKIP',
            'UNION', 'ALL', 'OPTIONAL', 'FOREACH', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'AND', 'OR', 'NOT', 'XOR', 'IN', 'STARTS',
            'ENDS', 'CONTAINS', 'IS', 'NULL', 'DISTINCT', 'AS', 'ASC', 'DESC',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COLLECT'
        }

    def normalize_whitespace_and_newlines(self, query: str) -> str:
        """Normalize all whitespace characters including tabs and newlines"""
        if not query:
            return ""

        # Replace all whitespace characters (spaces, tabs, newlines) with single space
        normalized = re.sub(r'\s+', ' ', query.strip())

        # Normalize around operators and punctuation
        normalized = re.sub(r'\s*([(){}[\],;])\s*', r'\1', normalized)
        normalized = re.sub(r'\s*([<>=!]+)\s*', r' \1 ', normalized)
        normalized = re.sub(r'\s*(->|<-)\s*', r'\1', normalized)  # Relationships

        return normalized.strip()

    def normalize_keywords_case(self, query: str) -> str:
        """Convert Cypher keywords to uppercase while preserving other identifiers"""
        if not query:
            return ""

        # Split by spaces but preserve quoted strings
        tokens = re.findall(r'["\'].*?["\']|\S+', query)
        normalized_tokens = []

        for token in tokens:
            # Skip quoted strings
            if token.startswith('"') or token.startswith("'"):
                normalized_tokens.append(token)
                continue

            # Check if token (without punctuation) is a Cypher keyword
            clean_token = re.sub(r'[(){}[\],;:.]+', '', token).upper()

            if clean_token in self.cypher_keywords:
                # Replace the keyword part with uppercase, keep punctuation
                pattern = re.escape(clean_token.lower())
                normalized_token = re.sub(pattern, clean_token, token, flags=re.IGNORECASE)
                normalized_tokens.append(normalized_token)
            else:
                # Keep original case for non-keywords
                normalized_tokens.append(token)

        return ' '.join(normalized_tokens)

    def extract_aliases(self, query: str) -> Dict[str, str]:
        """Extract alias mappings from query (e.g., 'n' AS 'node')"""
        aliases = {}

        # Pattern to match 'variable AS alias' or 'variable alias'
        as_patterns = [
            r'(\w+)\s+AS\s+(\w+)',  # explicit AS
            r'(\([^)]+\))\s+AS\s+(\w+)',  # function AS alias
            r'(\w+\.\w+)\s+AS\s+(\w+)',  # property AS alias
        ]

        for pattern in as_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                original, alias = match
                aliases[alias] = original

        return aliases

    def normalize_aliases(self, query1: str, query2: str) -> Tuple[str, str]:
        """Normalize aliases between two queries for fair comparison"""
        # Extract aliases from both queries
        aliases1 = self.extract_aliases(query1)
        aliases2 = self.extract_aliases(query2)

        # Create a mapping to standardize variable names
        var_mapping1 = {}
        var_mapping2 = {}
        var_counter = 1

        # Find all variables in both queries
        variables1 = set(re.findall(r'\b[a-z]\w*\b', query1, re.IGNORECASE))
        variables2 = set(re.findall(r'\b[a-z]\w*\b', query2, re.IGNORECASE))

        # Remove keywords from variables
        variables1 = {v for v in variables1 if v.upper() not in self.cypher_keywords}
        variables2 = {v for v in variables2 if v.upper() not in self.cypher_keywords}

        # Create standardized variable mappings
        all_vars = sorted(variables1.union(variables2))
        for var in all_vars:
            std_var = f"var{var_counter}"
            if var in variables1:
                var_mapping1[var] = std_var
            if var in variables2:
                var_mapping2[var] = std_var
            var_counter += 1

        # Apply mappings
        normalized1 = self._apply_variable_mapping(query1, var_mapping1)
        normalized2 = self._apply_variable_mapping(query2, var_mapping2)

        return normalized1, normalized2

    def _apply_variable_mapping(self, query: str, mapping: Dict[str, str]) -> str:
        """Apply variable name mapping to query"""
        result = query
        for old_var, new_var in mapping.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(old_var) + r'\b'
            result = re.sub(pattern, new_var, result)
        return result

    def remove_optional_elements(self, query: str) -> str:
        """Remove optional elements that don't affect query semantics"""
        if not query:
            return ""

        # Remove trailing semicolon
        query = query.rstrip(';')

        # Normalize quotes (single vs double)
        query = re.sub(r"'([^']*)'", r'"\1"', query)

        # Remove optional parentheses around single variables
        # query = re.sub(r'\((\w+)\)', r'\1', query)

        return query

    def normalize_query(self, query: str, level: str = 'full') -> str:
        """
        Comprehensive query normalization

        Args:
            query: Input Cypher query
            level: 'basic', 'standard', or 'full'
        """
        if not query:
            return ""

        normalized = query

        # Basic normalization - always apply
        normalized = self.normalize_whitespace_and_newlines(normalized)
        normalized = self.remove_optional_elements(normalized)

        if level in ['standard', 'full']:
            normalized = self.normalize_keywords_case(normalized)

        if level == 'full':
            # For full normalization, we need both queries to normalize aliases
            # This method is for single query, so we skip alias normalization here
            pass

        return normalized

    def normalize_query_pair(self, query1: str, query2: str) -> Tuple[str, str]:
        """Normalize a pair of queries with alias synchronization"""
        # First apply individual normalization
        norm1 = self.normalize_query(query1, 'standard')
        norm2 = self.normalize_query(query2, 'standard')

        # Then synchronize aliases
        norm1, norm2 = self.normalize_aliases(norm1, norm2)

        return norm1, norm2

    def calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between two Cypher queries"""
        norm1, norm2 = self.normalize_query_pair(query1, query2)

        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, norm1.lower(), norm2.lower())
        return matcher.ratio()

    def validate_cypher_syntax(self, query: str) -> Tuple[bool, str]:
        """
        Basic Cypher syntax validation

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query.strip():
            return False, "Empty query"

        # Basic syntax checks
        errors = []

        # Check for basic structure
        query_upper = query.upper()

        # Must have at least one main clause
        main_clauses = ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE']
        if not any(clause in query_upper for clause in main_clauses):
            errors.append("Query must contain at least one main clause (MATCH, CREATE, etc.)")

        # Check parentheses balance
        if query.count('(') != query.count(')'):
            errors.append("Unbalanced parentheses")

        # Check brackets balance
        if query.count('[') != query.count(']'):
            errors.append("Unbalanced square brackets")

        # Check braces balance
        if query.count('{') != query.count('}'):
            errors.append("Unbalanced curly braces")

        # Check quotes balance
        if query.count('"') % 2 != 0:
            errors.append("Unbalanced double quotes")
        if query.count("'") % 2 != 0:
            errors.append("Unbalanced single quotes")

        # Check for common syntax patterns
        # RETURN should come after MATCH/WITH
        if 'RETURN' in query_upper and 'MATCH' not in query_upper and 'WITH' not in query_upper:
            errors.append("RETURN clause without preceding MATCH or WITH")

        # Check for valid relationship patterns
        invalid_relationships = re.findall(r'-\[[^\]]*\]->', query)
        for rel in invalid_relationships:
            if not re.match(r'-\[:\w*\*?\d*\.\.\d*\]->', rel) and not re.match(r'-\[:\w+\]->', rel) and not re.match(r'-\[\]->', rel):
                errors.append(f"Invalid relationship syntax: {rel}")

        if errors:
            return False, "; ".join(errors)

        return True, "Valid syntax"


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
