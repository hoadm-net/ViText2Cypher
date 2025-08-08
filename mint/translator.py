"""
Translation utilities for translating English questions to Vietnamese using OpenAI API
"""

import openai
import json
import time
import os
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CypherTranslator:
    """Handle translation of English questions to Vietnamese for Cypher queries."""
    
    def __init__(self, template_file: str = "templates/translation_prompt.txt"):
        """
        Initialize CypherTranslator.
        
        Args:
            template_file: Path to the prompt template file
        """
        # Get API key from .env file
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        # Get configuration parameters from .env
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
            
        self.client = openai.OpenAI(api_key=api_key)
        self.template_file = template_file
        self.prompt_template = self._load_template()

    def _load_template(self) -> PromptTemplate:
        """Load prompt template from file using LangChain."""
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            return PromptTemplate(
                input_variables=["schema", "question"],
                template=template_content
            )
        except FileNotFoundError:
            print(f"Warning: Template file {self.template_file} not found. Using default template.")
            return self._get_default_template()

    def _get_default_template(self) -> PromptTemplate:
        """Default template if file not found."""
        default_template = """You are a professional translator specializing in translating natural language questions about graph databases from English to Vietnamese.

CONTEXT: You are translating questions that will be converted to Cypher queries for Neo4j graph databases.

IMPORTANT RULES:
1. Translate the question naturally and accurately into Vietnamese
2. DO NOT translate any database-related terms in quotes ("") - keep them exactly as they are
3. DO NOT translate property names, node labels, relationship types, or any technical identifiers
4. Maintain the technical meaning for graph database context
5. Use appropriate Vietnamese grammar and syntax
6. Keep the structure that allows for Cypher query generation

SCHEMA FOR THIS QUESTION:
{schema}

QUESTION TO TRANSLATE:
{question}

VIETNAMESE TRANSLATION:"""

        return PromptTemplate(
            input_variables=["schema", "question"],
            template=default_template
        )

    def get_translation_prompt(self, question: str, schema: str) -> str:
        """Create translation prompt with schema and question using LangChain."""
        return self.prompt_template.format(
            schema=schema or "No specific schema provided",
            question=question
        )

    def translate_question(self, question: str, schema: Optional[str] = None, max_retries: int = 3) -> str:
        """
        Translate a question from English to Vietnamese with specific schema.
        
        Args:
            question: English question to translate
            schema: Database schema for context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Vietnamese translation of the question
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in graph database queries translation from English to Vietnamese."
                        },
                        {
                            "role": "user",
                            "content": self.get_translation_prompt(question, schema)
                        }
                    ],
                    temperature=0.1,
                    max_tokens=self.max_tokens
                )

                translation = response.choices[0].message.content.strip()
                return translation

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"Translation failed: {question}"

        return question  # Return original if all attempts fail

    def translate_batch(self, samples: List[Dict[str, Any]], progress_desc: str = "Translating") -> List[Dict[str, Any]]:
        """
        Translate a batch of samples with progress tracking.
        
        Args:
            samples: List of samples to translate
            progress_desc: Description for progress bar
            
        Returns:
            List of translated samples
        """
        translated_data = []

        # Use tqdm to show progress bar
        for sample in tqdm(samples, desc=progress_desc, unit="question"):
            question = sample['question']
            schema = sample.get('schema', None)
            cypher = sample.get('cypher', None)

            if question and question.strip():
                translation = self.translate_question(question.strip(), schema)

                # Create result object
                result_item = {
                    'question_en': question,
                    'question_vi': translation,
                    'schema': schema,
                    'cypher': cypher
                }
                translated_data.append(result_item)

                # Short pause to avoid rate limit
                time.sleep(1)

        return translated_data

    def translate_and_save(self, input_file: str, output_file: str, start: int = 0, end: Optional[int] = None) -> None:
        """
        Translate questions from input file and save to output file.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            start: Start position
            end: End position (if None, goes to end of data)
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process data to extract question and schema
        samples = self._process_input_data(data)

        # Get data slice according to start and end
        if end is None:
            end = len(samples)

        samples_to_translate = samples[start:end]
        total = len(samples_to_translate)

        print(f"Translating {total} questions from JSON (from {start} to {end-1}) with specific schema...")

        # Translate samples
        translated_data = self.translate_batch(samples_to_translate)

        # Save final results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        print(f"Completed! Translated {len(translated_data)} questions and saved to {output_file}")

    def _process_input_data(self, data: Any) -> List[Dict[str, Any]]:
        """Process input data to extract questions and schemas."""
        samples = []
        if isinstance(data, list):
            for item in data:
                if 'question' in item:
                    sample = {
                        'question': item['question'],
                        'schema': item.get('schema', None),
                        'cypher': item.get('cypher', None)
                    }
                    samples.append(sample)
                elif 'text' in item:
                    sample = {
                        'question': item['text'],
                        'schema': item.get('schema', None),
                        'cypher': item.get('cypher', None)
                    }
                    samples.append(sample)
        elif isinstance(data, dict) and 'questions' in data:
            # Fallback for other formats
            for question in data['questions']:
                samples.append({
                    'question': question,
                    'schema': None,
                    'cypher': None
                })
        
        return samples
