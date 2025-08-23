#!/usr/bin/env python3
"""
Merge translated data files into a single organized file
Merge: translated_data_0_99.json, translated_data_100_1999.json, 
       translated_data_2000_2999.json, translated_data_3000_4437.json
Output: translated_data.json with fields: index, question, schema, cypher, translation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataMerger:
    def __init__(self):
        self.input_files = [
            "translated_data_0_99.json",
            "translated_data_100_1999.json", 
            "translated_data_2000_2999.json",
            "translated_data_3000_4437.json"
        ]
        self.output_file = "../data/translated_data.json"
        
    def load_translation_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load translations from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract translations array
            if "translations" in data:
                translations = data["translations"]
                logger.info(f"✅ Loaded {len(translations)} translations from {file_path}")
                return translations
            else:
                logger.warning(f"⚠️ No 'translations' key found in {file_path}")
                return []
                
        except FileNotFoundError:
            logger.warning(f"⚠️ File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return []
    
    def extract_index_from_question_id(self, item: Dict[str, Any]) -> int:
        """Extract index from question_id or sample_id field"""
        # Try different possible index fields
        for field in ["question_id", "sample_id", "index", "id"]:
            if field in item and isinstance(item[field], int):
                return item[field]
        
        # If no index field found, return -1 for manual handling
        return -1
    
    def normalize_translation_item(self, item: Dict[str, Any], file_index: int, item_index: int) -> Dict[str, Any]:
        """Normalize a translation item to standard format"""
        # Extract basic fields
        question = item.get("question", "")
        schema = item.get("schema", "")
        cypher = item.get("cypher", "")
        translation = item.get("translation", "")
        
        # Try to get original index
        original_index = self.extract_index_from_question_id(item)
        
        # If no original index, estimate based on file and position
        if original_index == -1:
            if "0_99" in self.input_files[file_index]:
                estimated_index = item_index
            elif "100_1999" in self.input_files[file_index]:
                estimated_index = 100 + item_index
            elif "2000_2999" in self.input_files[file_index]:
                estimated_index = 2000 + item_index
            elif "3000_4437" in self.input_files[file_index]:
                estimated_index = 3000 + item_index
            else:
                estimated_index = item_index
            
            logger.debug(f"Estimated index {estimated_index} for item in {self.input_files[file_index]}")
            original_index = estimated_index
        
        return {
            "index": original_index,
            "question": question,
            "schema": schema,
            "cypher": cypher,
            "translation": translation
        }
    
    def merge_all_files(self) -> List[Dict[str, Any]]:
        """Merge all translation files"""
        logger.info("🔄 Starting merge process...")
        
        all_translations = []
        total_loaded = 0
        
        for file_index, file_path in enumerate(self.input_files):
            logger.info(f"📖 Processing {file_path}")
            
            translations = self.load_translation_file(file_path)
            
            # Normalize each translation item
            for item_index, item in enumerate(translations):
                normalized_item = self.normalize_translation_item(item, file_index, item_index)
                all_translations.append(normalized_item)
            
            total_loaded += len(translations)
            logger.info(f"✅ Added {len(translations)} items from {file_path}")
        
        logger.info(f"📊 Total translations loaded: {total_loaded}")
        
        # Sort by index to ensure proper order
        all_translations.sort(key=lambda x: x["index"])
        logger.info("✅ Sorted translations by index")
        
        return all_translations
    
    def validate_merged_data(self, translations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the merged data and generate statistics"""
        if not translations:
            return {"valid": False, "error": "No translations found"}
        
        # Check for required fields
        required_fields = ["index", "question", "schema", "cypher", "translation"]
        
        valid_count = 0
        missing_fields = []
        empty_translations = 0
        index_gaps = []
        
        for i, item in enumerate(translations):
            # Check required fields
            item_valid = True
            for field in required_fields:
                if field not in item:
                    missing_fields.append(f"Item {i}: missing '{field}'")
                    item_valid = False
            
            if item_valid:
                valid_count += 1
            
            # Check for empty translations
            if not item.get("translation", "").strip():
                empty_translations += 1
        
        # Check for index gaps
        indices = [item["index"] for item in translations]
        if indices:
            min_idx, max_idx = min(indices), max(indices)
            expected_indices = set(range(min_idx, max_idx + 1))
            actual_indices = set(indices)
            missing_indices = expected_indices - actual_indices
            if missing_indices:
                index_gaps = sorted(list(missing_indices))
        
        stats = {
            "valid": len(missing_fields) == 0,
            "total_items": len(translations),
            "valid_items": valid_count,
            "empty_translations": empty_translations,
            "missing_fields": missing_fields[:10],  # Show first 10 errors
            "index_range": f"{min(indices)} - {max(indices)}" if indices else "N/A",
            "index_gaps": index_gaps[:20],  # Show first 20 gaps
            "has_duplicates": len(indices) != len(set(indices))
        }
        
        return stats
    
    def save_merged_data(self, translations: List[Dict[str, Any]]):
        """Save merged data to output file"""
        logger.info(f"💾 Saving merged data to {self.output_file}")
        
        # Create final output structure
        output_data = {
            "metadata": {
                "total_translations": len(translations),
                "source_files": self.input_files,
                "created_at": "2025-08-21",
                "description": "Merged translation data from multiple batch files"
            },
            "translations": translations
        }
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Successfully saved {len(translations)} translations to {self.output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save merged data: {e}")
            raise e
    
    def run_merge(self):
        """Run the complete merge process"""
        logger.info("🚀 Starting translation data merge")
        
        # Check input files exist
        missing_files = []
        for file_path in self.input_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"⚠️ Missing files: {missing_files}")
            logger.info("Continuing with available files...")
            self.input_files = [f for f in self.input_files if Path(f).exists()]
        
        if not self.input_files:
            logger.error("❌ No input files found!")
            return
        
        # Merge all files
        merged_translations = self.merge_all_files()
        
        if not merged_translations:
            logger.error("❌ No translations found in any file!")
            return
        
        # Validate merged data
        validation_stats = self.validate_merged_data(merged_translations)
        
        logger.info("📊 Validation Results:")
        logger.info(f"   - Total items: {validation_stats['total_items']}")
        logger.info(f"   - Valid items: {validation_stats['valid_items']}")
        logger.info(f"   - Empty translations: {validation_stats['empty_translations']}")
        logger.info(f"   - Index range: {validation_stats['index_range']}")
        
        if validation_stats['index_gaps']:
            logger.warning(f"   - Index gaps found: {len(validation_stats['index_gaps'])} gaps")
            logger.warning(f"   - First few gaps: {validation_stats['index_gaps'][:10]}")
        
        if validation_stats['has_duplicates']:
            logger.warning("   - ⚠️ Duplicate indices detected!")
        
        if not validation_stats['valid']:
            logger.warning("⚠️ Validation issues found, but continuing...")
            if validation_stats['missing_fields']:
                logger.warning(f"Missing fields: {validation_stats['missing_fields'][:5]}")
        
        # Save merged data
        self.save_merged_data(merged_translations)
        
        # Final summary
        logger.info("🎉 Merge completed successfully!")
        logger.info(f"📄 Output file: {self.output_file}")
        logger.info(f"📊 Final count: {len(merged_translations)} translations")

def main():
    """Main function"""
    merger = TranslationDataMerger()
    merger.run_merge()

if __name__ == "__main__":
    main()
