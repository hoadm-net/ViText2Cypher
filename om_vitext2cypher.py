#!/usr/bin/env python3
"""
Open Modal ViText2Cypher Pipeline with Qwen/Qwen2.5-7B
======================================================

Complete pipeline for Vietnamese Natural Language to Cypher translation using Qwen model.
Designed for Google Colab execution with Hugging Face Transformers.

Features:
- Load Vietnamese translated questions from translated_data.json
- Use Qwen/Qwen2.5-7B to generate Cypher queries
- Evaluate results using Variable-Agnostic evaluation v4.0
- Support for batch processing with start/end parameters
- Self-contained script for Colab execution

Author: ViText2Cypher Project
Date: August 25, 2025
"""

import json
import argparse
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from tqdm import tqdm
import torch

# Install requirements if needed (uncomment for Colab)
# !pip install transformers torch accelerate bitsandbytes

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
except ImportError:
    print("❌ Missing required libraries. Please install:")
    print("pip install transformers torch accelerate bitsandbytes")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# VARIABLE-AGNOSTIC EVALUATION
# ===============================

# Conservative semantic mappings for evaluation
PROPERTY_NAME_MAPPING = {
    'SCREEN_NAME': ['USERNAME', 'USER_NAME'],
    'USERNAME': ['SCREEN_NAME', 'USER_NAME'],
    'USER_NAME': ['SCREEN_NAME', 'USERNAME'],
    'ID': ['IDENTIFIER'],
    'IDENTIFIER': ['ID'],
    'UNITPRICE': ['UNIT_PRICE'],
    'UNIT_PRICE': ['UNITPRICE'],
    'COMPANYNAME': ['COMPANY_NAME'],
    'COMPANY_NAME': ['COMPANYNAME'],
    'CUSTOMERID': ['CUSTOMER_ID'],
    'CUSTOMER_ID': ['CUSTOMERID'],
    'ORDERID': ['ORDER_ID'],
    'ORDER_ID': ['ORDERID'],
    'PRODUCTID': ['PRODUCT_ID'],
    'PRODUCT_ID': ['PRODUCTID']
}

LABEL_MAPPING = {
    'USER': ['ME'],
    'ME': ['USER'],
}

@dataclass
class NormalizedPropertyFilter:
    """Property filter without variable names - only semantic content"""
    label_context: str  # Which label this property belongs to (e.g., 'MOVIE', 'USER')
    property_name: str  # Normalized property name
    operator: str
    value: str

    def __hash__(self):
        return hash((self.label_context, self.property_name, self.operator, self.value))

    def __eq__(self, other):
        return (isinstance(other, NormalizedPropertyFilter) and
                self.label_context == other.label_context and
                self.property_name == other.property_name and
                self.operator == other.operator and
                self.value == other.value)

@dataclass
class NormalizedPattern:
    """Pattern representation without variable names"""
    node_labels: List[str]  # Sorted list of normalized node labels
    relationship_types: List[str]  # Sorted list of relationship types
    inline_filters: Set[NormalizedPropertyFilter]  # Property filters from inline properties

    def __hash__(self):
        return hash((tuple(self.node_labels), tuple(self.relationship_types), frozenset(self.inline_filters)))

@dataclass
class NormalizedCypherComponents:
    """Completely normalized components without variable dependencies"""
    patterns: List[NormalizedPattern]
    where_filters: Set[NormalizedPropertyFilter]
    return_aggregations: Set[str]  # Normalized aggregation functions
    return_properties: Set[str]  # Normalized property accesses
    order_by_items: Set[str]  # Normalized order items
    limit_skip: Dict[str, int]
    keywords: Set[str]

    def to_dict(self):
        return {
            'patterns': [self._pattern_to_dict(p) for p in self.patterns],
            'where_filters': [self._filter_to_dict(f) for f in self.where_filters],
            'return_aggregations': list(self.return_aggregations),
            'return_properties': list(self.return_properties),
            'order_by_items': list(self.order_by_items),
            'limit_skip': self.limit_skip,
            'keywords': list(self.keywords)
        }

    def _pattern_to_dict(self, pattern: NormalizedPattern):
        return {
            'node_labels': pattern.node_labels,
            'relationship_types': pattern.relationship_types,
            'inline_filters': [self._filter_to_dict(f) for f in pattern.inline_filters]
        }

    def _filter_to_dict(self, filter_obj: NormalizedPropertyFilter):
        return {
            'label_context': filter_obj.label_context,
            'property': filter_obj.property_name,
            'operator': filter_obj.operator,
            'value': filter_obj.value
        }

class VariableAgnosticNormalizer:
    """Normalizer that completely ignores variable names"""

    def __init__(self):
        # Patterns for parsing
        self.match_pattern = re.compile(r'MATCH\s+(.+?)(?=\s+(?:WHERE|WITH|RETURN|ORDER|LIMIT|$))', re.IGNORECASE | re.DOTALL)
        self.where_pattern = re.compile(r'WHERE\s+(.+?)(?=\s+(?:WITH|RETURN|ORDER|LIMIT|$))', re.IGNORECASE | re.DOTALL)
        self.with_pattern = re.compile(r'WITH\s+(.+?)(?=\s+(?:WHERE|RETURN|ORDER|LIMIT|MATCH|$))', re.IGNORECASE | re.DOTALL)
        self.return_pattern = re.compile(r'RETURN\s+(.+?)(?=\s+(?:ORDER|LIMIT|$))', re.IGNORECASE | re.DOTALL)
        self.order_pattern = re.compile(r'ORDER\s+BY\s+(.+?)(?=\s+(?:LIMIT|$))', re.IGNORECASE | re.DOTALL)
        self.limit_pattern = re.compile(r'LIMIT\s+(\d+)', re.IGNORECASE)
        self.skip_pattern = re.compile(r'SKIP\s+(\d+)', re.IGNORECASE)

    def parse_normalized_components(self, cypher: str) -> NormalizedCypherComponents:
        """Parse Cypher into completely normalized components"""
        cypher_norm = self._basic_normalize(cypher)

        # Parse patterns and normalize them
        patterns = self._parse_and_normalize_patterns(cypher_norm)

        # Parse WHERE filters and normalize them
        where_filters = self._parse_and_normalize_where_filters(cypher_norm)

        # Parse return items and normalize them
        return_aggs, return_props = self._parse_and_normalize_return_items(cypher_norm)

        # Parse other components
        order_by = self._parse_and_normalize_order_by(cypher_norm)
        limit_skip = self._parse_limit_skip(cypher_norm)
        keywords = self._parse_keywords(cypher_norm)

        return NormalizedCypherComponents(
            patterns=patterns,
            where_filters=where_filters,
            return_aggregations=return_aggs,
            return_properties=return_props,
            order_by_items=order_by,
            limit_skip=limit_skip,
            keywords=keywords
        )

    def _basic_normalize(self, cypher: str) -> str:
        """Basic text normalization"""
        if not cypher or cypher.strip() == '':
            return ''

        # Fix escaped newlines issue: convert \\n to actual newlines first
        cypher = cypher.replace('\\n', '\n')

        cypher = re.sub(r'\s+', ' ', cypher.strip())
        return cypher.upper()

    def _parse_and_normalize_patterns(self, cypher: str) -> List[NormalizedPattern]:
        """Parse MATCH patterns and create variable-agnostic representations"""
        patterns = []

        for match in self.match_pattern.finditer(cypher):
            pattern_str = match.group(1).strip()
            normalized_pattern = self._normalize_single_pattern(pattern_str)
            if normalized_pattern:
                patterns.append(normalized_pattern)

        return patterns

    def _normalize_single_pattern(self, pattern_str: str) -> Optional[NormalizedPattern]:
        """Convert a single pattern to normalized form"""
        # Extract all node labels (ignore variable names)
        node_labels = []
        node_pattern = re.compile(r'\(\w*:(\w+)(?:\s*\{[^}]*\})?\)', re.IGNORECASE)
        for match in node_pattern.finditer(pattern_str):
            label = self._normalize_label(match.group(1))
            node_labels.append(label)

        # Extract all relationship types (ignore variable names)
        relationship_types = []
        # Match both -[var:TYPE]- and -[:TYPE]- patterns
        rel_pattern = re.compile(r'-\[(?:\w*:)?(\w+)(?:\s*\{[^}]*\})?\]-', re.IGNORECASE)
        for match in rel_pattern.finditer(pattern_str):
            rel_type = match.group(1).upper()
            relationship_types.append(rel_type)

        # Extract inline property filters
        inline_filters = self._extract_inline_filters(pattern_str)

        if not node_labels and not relationship_types:
            return None

        # Sort for consistent comparison
        node_labels.sort()
        relationship_types.sort()

        return NormalizedPattern(
            node_labels=node_labels,
            relationship_types=relationship_types,
            inline_filters=inline_filters
        )

    def _extract_inline_filters(self, pattern_str: str) -> Set[NormalizedPropertyFilter]:
        """Extract property filters from inline properties like {name: 'value'}"""
        filters = set()

        # Find all property blocks {prop: value, prop2: value2}
        prop_blocks = re.findall(r'\{([^}]+)\}', pattern_str, re.IGNORECASE)

        for block in prop_blocks:
            # Find the associated label for this property block
            # Look for the pattern before this block
            label_match = re.search(r':(\w+)\s*\{[^}]*' + re.escape(block), pattern_str, re.IGNORECASE)
            if label_match:
                label = self._normalize_label(label_match.group(1))

                # Parse property: value pairs within this block
                prop_pairs = re.findall(r'(\w+):\s*([\'"][^\'\"]*[\'"]|\d+)', block, re.IGNORECASE)

                for prop_name, value in prop_pairs:
                    normalized_prop = self._normalize_property_name(prop_name)
                    clean_value = value.strip('\'"')

                    filter_obj = NormalizedPropertyFilter(
                        label_context=label,
                        property_name=normalized_prop,
                        operator='=',
                        value=f"'{clean_value}'" if not value.isdigit() else clean_value
                    )
                    filters.add(filter_obj)

        return filters

    def _parse_and_normalize_where_filters(self, cypher: str) -> Set[NormalizedPropertyFilter]:
        """Parse WHERE clause and create normalized filters"""
        filters = set()

        for match in self.where_pattern.finditer(cypher):
            where_clause = match.group(1).strip()

            # Create a variable to label mapping for this query
            var_to_label = self._create_variable_to_label_mapping(cypher)

            # Extract property filters like var.prop = value
            prop_filter_pattern = re.compile(r'(\w+)\.(\w+)\s*([=<>!]+)\s*([\'"][^\'\"]*[\'"]|\d+)', re.IGNORECASE)

            for filter_match in prop_filter_pattern.finditer(where_clause):
                var_name = filter_match.group(1)
                prop_name = filter_match.group(2)
                operator = filter_match.group(3)
                value = filter_match.group(4)

                # Map variable to its label
                label_context = var_to_label.get(var_name, 'UNKNOWN')

                filter_obj = NormalizedPropertyFilter(
                    label_context=label_context,
                    property_name=self._normalize_property_name(prop_name),
                    operator=operator,
                    value=value
                )
                filters.add(filter_obj)

        return filters

    def _create_variable_to_label_mapping(self, cypher: str) -> Dict[str, str]:
        """Create mapping from variable names to their labels"""
        var_to_label = {}

        # Find all MATCH patterns and extract variable -> label mappings
        for match in self.match_pattern.finditer(cypher):
            pattern_str = match.group(1).strip()

            # Extract (var:Label) patterns
            node_pattern = re.compile(r'\((\w+):(\w+)(?:\s*\{[^}]*\})?\)', re.IGNORECASE)
            for node_match in node_pattern.finditer(pattern_str):
                var_name = node_match.group(1)
                label = self._normalize_label(node_match.group(2))
                var_to_label[var_name] = label

        return var_to_label

    def _parse_and_normalize_return_items(self, cypher: str) -> Tuple[Set[str], Set[str]]:
        """Parse RETURN items and separate aggregations from properties"""
        aggregations = set()
        properties = set()

        for match in self.return_pattern.finditer(cypher):
            return_part = match.group(1).strip()
            return_items = [item.strip() for item in return_part.split(',')]

            for item in return_items:
                # Remove AS aliases
                item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.IGNORECASE).strip()

                # Check if it's an aggregation function
                if any(agg in item.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
                    normalized_agg = self._normalize_aggregation_function(item)
                    aggregations.add(normalized_agg)
                else:
                    # It's a property access - normalize it
                    normalized_prop = self._normalize_property_access(item)
                    properties.add(normalized_prop)

        return aggregations, properties

    def _normalize_aggregation_function(self, func_str: str) -> str:
        """Normalize aggregation functions ignoring variable names"""
        func_str = func_str.upper().strip()

        # COUNT(*) or COUNT(anything) -> COUNT_ALL
        if re.match(r'COUNT\s*\([^)]*\)', func_str):
            return 'COUNT_ALL'

        # Handle property aggregations like AVG(var.prop) -> AVG_PROP
        agg_prop_match = re.match(r'(AVG|SUM|MIN|MAX)\s*\(\s*\w+\.(\w+)\s*\)', func_str)
        if agg_prop_match:
            agg_func = agg_prop_match.group(1)
            prop_name = self._normalize_property_name(agg_prop_match.group(2))
            return f'{agg_func}_{prop_name}'

        # Handle simple aggregations like AVG(prop) -> AVG_PROP
        simple_agg_match = re.match(r'(AVG|SUM|MIN|MAX)\s*\(\s*(\w+)\s*\)', func_str)
        if simple_agg_match:
            agg_func = simple_agg_match.group(1)
            prop_name = self._normalize_property_name(simple_agg_match.group(2))
            return f'{agg_func}_{prop_name}'

        return func_str

    def _normalize_property_access(self, item: str) -> str:
        """Normalize property access ignoring variable names"""
        # Handle var.property -> property
        prop_match = re.match(r'\w+\.(\w+)', item, re.IGNORECASE)
        if prop_match:
            prop_name = prop_match.group(1)
            return self._normalize_property_name(prop_name)

        # Handle simple property names
        if re.match(r'^\w+$', item):
            return self._normalize_property_name(item)

        return item.upper()

    def _parse_and_normalize_order_by(self, cypher: str) -> Set[str]:
        """Parse ORDER BY clause and normalize items"""
        order_items = set()

        for match in self.order_pattern.finditer(cypher):
            order_part = match.group(1).strip()
            items = [item.strip() for item in order_part.split(',')]

            for item in items:
                # Remove DESC/ASC and normalize the property
                clean_item = re.sub(r'\s+(ASC|DESC)$', '', item, flags=re.IGNORECASE).strip()
                normalized_item = self._normalize_property_access(clean_item)
                order_items.add(normalized_item)

        return order_items

    def _parse_limit_skip(self, cypher: str) -> Dict[str, int]:
        """Parse LIMIT/SKIP - no normalization needed"""
        limit_skip = {}

        limit_match = self.limit_pattern.search(cypher)
        if limit_match:
            limit_skip['LIMIT'] = int(limit_match.group(1))

        skip_match = self.skip_pattern.search(cypher)
        if skip_match:
            limit_skip['SKIP'] = int(skip_match.group(1))

        return limit_skip

    def _parse_keywords(self, cypher: str) -> Set[str]:
        """Parse special keywords"""
        keywords = set()

        if 'DISTINCT' in cypher:
            keywords.add('DISTINCT')
        if 'OPTIONAL MATCH' in cypher:
            keywords.add('OPTIONAL_MATCH')
        if 'EXISTS' in cypher:
            keywords.add('EXISTS')
        if 'NOT EXISTS' in cypher:
            keywords.add('NOT_EXISTS')

        return keywords

    def _normalize_property_name(self, prop_name: str) -> str:
        """Normalize property name using semantic mapping"""
        prop_name = prop_name.upper()

        for canonical, equivalents in PROPERTY_NAME_MAPPING.items():
            if prop_name == canonical or prop_name in equivalents:
                return canonical

        return prop_name

    def _normalize_label(self, label: str) -> str:
        """Normalize label using semantic mapping"""
        label = label.upper()

        for canonical, equivalents in LABEL_MAPPING.items():
            if label == canonical or label in equivalents:
                return canonical

        return label

class VariableAgnosticF1Calculator:
    """F1 calculator that ignores variable names completely"""

    @staticmethod
    def calculate_pattern_f1(patterns1: List[NormalizedPattern], patterns2: List[NormalizedPattern]) -> Tuple[float, float, float]:
        """Calculate F1 for patterns using variable-agnostic comparison"""
        if not patterns1 and not patterns2:
            return 1.0, 1.0, 1.0

        if not patterns1:
            return 1.0, 0.0, 0.0

        if not patterns2:
            return 0.0, 1.0, 0.0

        # Convert patterns to hashable representations for comparison
        patterns1_set = set(patterns1)
        patterns2_set = set(patterns2)

        intersection = patterns1_set.intersection(patterns2_set)

        precision = len(intersection) / len(patterns1_set) if patterns1_set else 0.0
        recall = len(intersection) / len(patterns2_set) if patterns2_set else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def calculate_filter_f1(filters1: Set[NormalizedPropertyFilter], filters2: Set[NormalizedPropertyFilter]) -> Tuple[float, float, float]:
        """Calculate F1 for normalized property filters"""
        if not filters1 and not filters2:
            return 1.0, 1.0, 1.0

        if not filters1:
            return 1.0, 0.0, 0.0

        if not filters2:
            return 0.0, 1.0, 0.0

        intersection = filters1.intersection(filters2)

        precision = len(intersection) / len(filters1)
        recall = len(intersection) / len(filters2)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def calculate_set_f1(set1: Set[str], set2: Set[str]) -> Tuple[float, float, float]:
        """Calculate F1 for any set of strings"""
        if not set1 and not set2:
            return 1.0, 1.0, 1.0

        if not set1:
            return 1.0, 0.0, 0.0

        if not set2:
            return 0.0, 1.0, 0.0

        intersection = set1.intersection(set2)

        precision = len(intersection) / len(set1)
        recall = len(intersection) / len(set2)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

class VariableAgnosticEvaluator:
    """Main evaluator that ignores variable names"""

    def __init__(self):
        self.normalizer = VariableAgnosticNormalizer()
        self.calculator = VariableAgnosticF1Calculator()

    def evaluate_cypher_pair(self, predicted: str, expected: str) -> Dict[str, Any]:
        """Evaluate a pair of Cypher queries with complete variable name agnostic comparison"""
        try:
            # Parse both queries
            pred_components = self.normalizer.parse_normalized_components(predicted)
            exp_components = self.normalizer.parse_normalized_components(expected)

            # Calculate F1 scores for each component
            scores = {}

            # Pattern matching F1
            pattern_p, pattern_r, pattern_f1 = self.calculator.calculate_pattern_f1(
                pred_components.patterns, exp_components.patterns
            )
            scores['pattern'] = {'precision': pattern_p, 'recall': pattern_r, 'f1': pattern_f1}

            # WHERE filters F1
            where_p, where_r, where_f1 = self.calculator.calculate_filter_f1(
                pred_components.where_filters, exp_components.where_filters
            )
            scores['where_filters'] = {'precision': where_p, 'recall': where_r, 'f1': where_f1}

            # Return aggregations F1
            agg_p, agg_r, agg_f1 = self.calculator.calculate_set_f1(
                pred_components.return_aggregations, exp_components.return_aggregations
            )
            scores['return_aggregations'] = {'precision': agg_p, 'recall': agg_r, 'f1': agg_f1}

            # Return properties F1
            prop_p, prop_r, prop_f1 = self.calculator.calculate_set_f1(
                pred_components.return_properties, exp_components.return_properties
            )
            scores['return_properties'] = {'precision': prop_p, 'recall': prop_r, 'f1': prop_f1}

            # Order by F1
            order_p, order_r, order_f1 = self.calculator.calculate_set_f1(
                pred_components.order_by_items, exp_components.order_by_items
            )
            scores['order_by'] = {'precision': order_p, 'recall': order_r, 'f1': order_f1}

            # Keywords F1
            kw_p, kw_r, kw_f1 = self.calculator.calculate_set_f1(
                pred_components.keywords, exp_components.keywords
            )
            scores['keywords'] = {'precision': kw_p, 'recall': kw_r, 'f1': kw_f1}

            # Limit/Skip exact match
            limit_skip_match = pred_components.limit_skip == exp_components.limit_skip
            scores['limit_skip'] = {'exact_match': limit_skip_match}

            # Calculate overall F1 score (weighted average)
            component_weights = {
                'pattern': 0.3,
                'where_filters': 0.25,
                'return_aggregations': 0.15,
                'return_properties': 0.15,
                'order_by': 0.1,
                'keywords': 0.05
            }

            overall_f1 = 0.0
            for component, weight in component_weights.items():
                if component in scores and 'f1' in scores[component]:
                    overall_f1 += scores[component]['f1'] * weight

            # Apply penalty for limit/skip mismatch
            if not limit_skip_match:
                overall_f1 *= 0.9

            scores['overall'] = {'f1': overall_f1}

            return {
                'scores': scores,
                'predicted_components': pred_components.to_dict(),
                'expected_components': exp_components.to_dict()
            }

        except Exception as e:
            logger.error(f"Error evaluating Cypher pair: {e}")
            return {
                'scores': {'overall': {'f1': 0.0}},
                'error': str(e)
            }

# ===============================
# QWEN MODEL HANDLER
# ===============================

class QwenCypherGenerator:
    """Qwen model handler for Cypher generation"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing Qwen model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"4-bit quantization: {use_4bit}")

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Qwen model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization if requested
            if self.use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            logger.info("✅ Qwen model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            raise

    def _create_prompt(self, vietnamese_question: str, schema: str) -> str:
        """Create Vietnamese NL2Cypher prompt for Qwen"""

        # Vietnamese prompt template optimized for Qwen
        template = """Bạn là một chuyên gia về cơ sở dữ liệu Neo4j và ngôn ngữ truy vấn Cypher. 
Nhiệm vụ của bạn là chuyển đổi câu hỏi tiếng Việt thành truy vấn Cypher chính xác.

Schema cơ sở dữ liệu:
{schema}

Câu hỏi tiếng Việt: {question}

Hãy tạo truy vấn Cypher chính xác cho câu hỏi trên. Chỉ trả về truy vấn Cypher, không có giải thích thêm.

Truy vấn Cypher:"""

        return template.format(question=vietnamese_question, schema=schema)

    def generate_cypher(self, vietnamese_question: str, schema: str) -> str:
        """Generate Cypher query from Vietnamese question"""
        try:
            # Create prompt
            prompt = self._create_prompt(vietnamese_question, schema)

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )

            # Clean up response
            cypher_query = generated_text.strip()

            # Remove any remaining markdown formatting
            if cypher_query.startswith("```"):
                lines = cypher_query.split('\n')
                cypher_query = '\n'.join([line for line in lines if not line.startswith("```")])

            # Remove common prefixes
            for prefix in ["Cypher:", "Query:", "Truy vấn:", "Answer:"]:
                if cypher_query.startswith(prefix):
                    cypher_query = cypher_query[len(prefix):].strip()

            return cypher_query.strip()

        except Exception as e:
            logger.error(f"Error generating Cypher with Qwen: {e}")
            return ""

# ===============================
# MAIN PIPELINE
# ===============================

class OMViText2CypherPipeline:
    """Complete pipeline for Vietnamese NL to Cypher translation using Qwen"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", use_4bit: bool = True):
        self.model_name = model_name
        self.qwen_generator = QwenCypherGenerator(model_name, use_4bit)
        self.evaluator = VariableAgnosticEvaluator()

    def _load_data(self, start: int = 0, end: Optional[int] = None) -> List[Dict]:
        """Load translated data from JSON file"""
        data_path = "translated_data.json"

        try:
            logger.info(f"Loading data from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ Data file not found: {data_path}")
            logger.info("Please ensure translated_data.json is in the same directory")
            raise

        # Handle slicing
        if end is None:
            end = len(data)

        selected_data = data[start:end]
        logger.info(f"Selected {len(selected_data)} samples (index {start} to {end-1})")

        return selected_data

    def process_batch(self, start: int = 0, end: Optional[int] = None) -> Dict[str, Any]:
        """Process a batch of Vietnamese questions and evaluate results"""

        # Load data
        data = self._load_data(start, end)

        results = []
        total_samples = len(data)

        logger.info(f"Starting OM-ViText2Cypher pipeline with Qwen model")
        logger.info(f"Processing {total_samples} samples...")

        for i, item in enumerate(tqdm(data, desc="Processing samples", unit="sample")):
            try:
                # Extract data
                index = item.get('index', start + i)
                vietnamese_question = item.get('translation', '')
                schema = item.get('schema', '')
                ground_truth_cypher = item.get('cypher', '')
                original_question = item.get('question', '')

                if not vietnamese_question or not schema or not ground_truth_cypher:
                    logger.warning(f"Missing data for sample {index}, skipping")
                    continue

                # Generate Cypher query using Qwen
                generated_cypher = self.qwen_generator.generate_cypher(vietnamese_question, schema)

                if not generated_cypher:
                    logger.warning(f"Failed to generate Cypher for sample {index}")
                    continue

                # Evaluate generated query against ground truth
                evaluation = self.evaluator.evaluate_cypher_pair(
                    generated_cypher,
                    ground_truth_cypher
                )

                # Store result
                result = {
                    'index': index,
                    'original_question': original_question,
                    'vietnamese_question': vietnamese_question,
                    'schema': schema,
                    'ground_truth_cypher': ground_truth_cypher,
                    'generated_cypher': generated_cypher,
                    'evaluation': evaluation
                }

                results.append(result)

                # Log progress for every 10 samples
                if (i + 1) % 10 == 0:
                    f1_score = evaluation['scores']['overall']['f1']
                    logger.info(f"Processed {i + 1}/{total_samples} - Latest F1: {f1_score:.3f}")

            except Exception as e:
                logger.error(f"Error processing sample {index}: {e}")
                continue

        # Calculate overall statistics
        if results:
            # Extract F1 scores from Variable-Agnostic evaluation format
            f1_scores = [r['evaluation']['scores']['overall']['f1'] for r in results if 'scores' in r['evaluation']]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

            # Count perfect matches (F1 = 1.0)
            perfect_matches = sum(1 for score in f1_scores if score == 1.0)

            # Component-wise F1 scores from Variable-Agnostic format
            component_f1s = {}
            component_names = ['pattern', 'where_filters', 'return_aggregations', 'return_properties', 'order_by', 'keywords']
            for component in component_names:
                component_scores = []
                for r in results:
                    if 'scores' in r['evaluation'] and component in r['evaluation']['scores']:
                        if 'f1' in r['evaluation']['scores'][component]:
                            component_scores.append(r['evaluation']['scores'][component]['f1'])
                if component_scores:
                    component_f1s[component] = sum(component_scores) / len(component_scores)
        else:
            perfect_matches = 0
            avg_f1 = 0.0
            component_f1s = {}

        statistics = {
            'total_samples_processed': len(results),
            'exact_match_count': perfect_matches,
            'exact_match_rate': perfect_matches / len(results) if results else 0.0,
            'average_f1_score': avg_f1,
            'component_f1_scores': component_f1s
        }

        return {
            'metadata': {
                'model_type': 'qwen',
                'model_name': self.model_name,
                'start_index': start,
                'end_index': end if end else len(data) + start,
                'processing_timestamp': datetime.now().isoformat(),
                'total_samples_in_range': total_samples,
                'successfully_processed': len(results)
            },
            'statistics': statistics,
            'detailed_results': results
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='OM-ViText2Cypher Pipeline with Qwen')
    parser.add_argument('--start', type=int, default=0, help='Start index (default: 0)')
    parser.add_argument('--end', type=int, help='End index (default: end of dataset)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B', help='Qwen model name')
    parser.add_argument('--use_4bit', action='store_true', default=True, help='Use 4-bit quantization')
    parser.add_argument('--output', type=str, help='Output file path (optional, will auto-generate if not provided)')

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = OMViText2CypherPipeline(model_name=args.model, use_4bit=args.use_4bit)

        # Determine actual end index
        try:
            with open('translated_data.json', 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            actual_end = args.end if args.end is not None else len(test_data)
        except FileNotFoundError:
            actual_end = args.end if args.end is not None else 100  # fallback

        # Auto-generate output filename if not provided
        if args.output is None:
            output_file = f'om_vitext2cypher_qwen_{args.start}_{actual_end}.json'
        else:
            output_file = args.output

        # Process batch
        results = pipeline.process_batch(start=args.start, end=actual_end)

        # Save results
        pipeline.save_results(results, output_file)

        # Print summary
        stats = results['statistics']
        logger.info("=== OM-VITEXT2CYPHER PIPELINE SUMMARY ===")
        logger.info(f"Model: {args.model}")
        logger.info(f"4-bit quantization: {args.use_4bit}")
        logger.info(f"Samples processed: {stats['total_samples_processed']}")
        logger.info(f"Exact match rate: {stats['exact_match_rate']:.4f}")
        logger.info(f"Average F1 score: {stats['average_f1_score']:.4f}")
        logger.info("Component F1 scores:")
        for component, f1 in stats['component_f1_scores'].items():
            logger.info(f"  {component}: {f1:.4f}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # For Colab usage, you can also run directly like this:
    # python om_vitext2cypher.py --start 0 --end 20 --model Qwen/Qwen2.5-7B --use_4bit
    main()
