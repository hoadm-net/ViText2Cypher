#!/usr/bin/env python3
"""
Data Augmentation Pipeline for ViText2Cypher
Thực hiện việc làm giàu dữ liệu bằng cách dịch tự động với few-shot learning
Sử dụng KNN để tìm examplars từ bản dịch thủ công
"""

import json
import os
import random
import numpy as np
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

class DataAugmentationPipeline:
    def __init__(self, debug: bool = True):
        """Khởi tạo pipeline với các cấu hình từ .env"""
        self.debug = debug
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))

        if self.debug:
            print(f"🔧 Config loaded:")
            print(f"   API Key: {'***' + self.openai_api_key[-10:] if self.openai_api_key else 'NOT FOUND'}")
            print(f"   Model: {self.openai_model}")
            print(f"   Embedding Model: {self.embedding_model}")

        # Khởi tạo OpenAI embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model=self.embedding_model
            )
            if self.debug:
                print("✓ OpenAI Embeddings initialized")
        except Exception as e:
            print(f"✗ Error initializing embeddings: {e}")
            raise

        # Khởi tạo OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Dữ liệu sẽ được load
        self.manual_translation_data = []
        self.train_data = []
        self.manual_embeddings = []

        # Load template từ file
        self.translation_template = self._load_template()

    def _load_template(self) -> PromptTemplate:
        """Load template từ file"""
        template_file = '../templates/few_shot_translation.txt'
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()

            if self.debug:
                print(f"✓ Template loaded from {template_file}")

            return PromptTemplate(
                input_variables=["schema", "examples", "question"],
                template=template_content
            )
        except Exception as e:
            print(f"✗ Error loading template: {e}")
            print(f"Creating default template...")
            # Fallback template
            default_template = """Bạn là chuyên gia dịch thuật chuyên dịch câu hỏi từ tiếng Anh sang tiếng Việt.

Schema: {schema}

Examples: {examples}

Question: {question}

Vietnamese:"""
            return PromptTemplate(
                input_variables=["schema", "examples", "question"],
                template=default_template
            )

    def load_manual_translation(self, file_path: str = '../data/manual_translation.json'):
        """Load manual translation data - sử dụng cache nếu có"""
        try:
            # Kiểm tra xem có file cache với embeddings không
            cache_file = file_path.replace('.json', '_with_embeddings.json')
            
            if os.path.exists(cache_file):
                if self.debug:
                    print(f"🔄 Loading from cache: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.manual_translation_data = json.load(f)
                
                # Load embeddings từ cache
                self.manual_embeddings = [item['embedding_question'] for item in self.manual_translation_data]
                if self.debug:
                    print(f"✓ Loaded {len(self.manual_translation_data)} samples with embeddings from cache")
                return True
            
            # Nếu không có cache, load file gốc
            with open(file_path, 'r', encoding='utf-8') as f:
                self.manual_translation_data = json.load(f)

            # Kiểm tra embeddings trong file gốc
            has_embeddings = any('embedding_question' in item for item in self.manual_translation_data)
            if has_embeddings:
                if self.debug:
                    print(f"✓ Đã load {len(self.manual_translation_data)} mẫu từ {file_path}")
                # Load embeddings
                self.manual_embeddings = [item['embedding_question'] for item in self.manual_translation_data]
                return True
            else:
                if self.debug:
                    print(f"⚠️ File không có embeddings - cần tạo cache embeddings...")
                    print(f"💡 Để tiết kiệm cost, hãy tạo cache trước:")
                    print(f"   python create_embeddings_cache.py")
                return False

        except Exception as e:
            print(f"✗ Lỗi khi load {file_path}: {e}")
            return False

    def _create_embeddings_for_manual_data(self) -> bool:
        """Tự tạo embeddings cho manual translation data"""
        try:
            if self.debug:
                print("🔄 Đang tạo embeddings cho manual translation data...")
            
            self.manual_embeddings = []
            for i, item in enumerate(self.manual_translation_data):
                if self.debug and i % 100 == 0:
                    print(f"   Đang xử lý {i}/{len(self.manual_translation_data)}...")
                
                embedding = self.embeddings.embed_query(item['question'])
                self.manual_embeddings.append(embedding)
                item['embedding_question'] = embedding
                
                # Delay nhỏ để tránh rate limit
                time.sleep(0.1)
            
            if self.debug:
                print(f"✓ Đã tạo {len(self.manual_embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"✗ Lỗi tạo embeddings: {e}")
            return False

    def load_data(self):
        """Load dữ liệu từ các file JSON"""
        if self.debug:
            print("🔄 Đang load dữ liệu...")

        # Load manual translation với embeddings
        if not self.load_manual_translation():
            print("❌ Không thể load manual translation data")
            return False

        # Hiển thị cấu trúc
        if self.debug and self.manual_translation_data:
            sample = self.manual_translation_data[0]
            print(f"   Cấu trúc mẫu: {list(sample.keys())}")

            # Kiểm tra trường translation
            translation_field = None
            if 'translation' in sample:
                translation_field = 'translation'
            elif 'question_vi' in sample:
                translation_field = 'question_vi'

            if translation_field:
                print(f"   Trường dịch: '{translation_field}'")
            else:
                print("   ⚠️ Không tìm thấy trường dịch")

        # Load train data
        try:
            with open('../data/train.json', 'r', encoding='utf-8') as f:
                self.train_data = json.load(f)
            if self.debug:
                print(f"✓ Đã load {len(self.train_data)} mẫu từ train.json")
        except Exception as e:
            print(f"✗ Lỗi load train.json: {e}")
            return False

        return True

    def find_similar_examples(self, target_question: str, k: int = 2) -> List[Dict[str, Any]]:
        """Tìm k mẫu tương tự nhất bằng KNN với cosine similarity"""
        try:
            # Tạo embedding cho câu hỏi target
            target_embedding = self.embeddings.embed_query(target_question)

            # Tính similarity với tất cả embeddings
            similarities = []
            for i, manual_embedding in enumerate(self.manual_embeddings):
                similarity = cosine_similarity(
                    [target_embedding],
                    [manual_embedding]
                )[0][0]
                similarities.append((i, similarity))

            # Sắp xếp theo độ tương đồng giảm dần
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Lấy k mẫu tương tự nhất
            top_k_indices = [idx for idx, _ in similarities[:k]]
            examples = [self.manual_translation_data[i] for i in top_k_indices]

            if self.debug:
                similarities_scores = [sim for _, sim in similarities[:k]]
                print(f"   📊 Top {k} similarity scores: {similarities_scores}")

            return examples

        except Exception as e:
            if self.debug:
                print(f"✗ Lỗi tìm examples: {e}")
            # Fallback: random examples
            return random.sample(self.manual_translation_data, min(k, len(self.manual_translation_data)))

    def format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples thành string cho few-shot prompt"""
        example_strings = []

        for i, example in enumerate(examples, 1):
            # Lấy translation từ trường có sẵn
            translation = example.get('translation') or example.get('question_vi', '[Cần dịch]')

            example_str = f"""
Example {i}:
English: {example['question']}
Vietnamese: {translation}
Schema: {example['schema'][:200]}...
Cypher: {example['cypher']}
"""
            example_strings.append(example_str)

        return "\n".join(example_strings)

    def translate_question(self, question: str, schema: str, examples: List[Dict[str, Any]]) -> str:
        """Dịch câu hỏi sử dụng few-shot learning"""
        try:
            # Format examples
            examples_text = self.format_examples(examples)

            # Tạo prompt từ template
            prompt = self.translation_template.format(
                schema=schema[:500] + "..." if len(schema) > 500 else schema,  # Rút gọn schema
                examples=examples_text,
                question=question
            )

            # Gọi OpenAI API
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia dịch thuật chuyên nghiệp."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )

            translation = response.choices[0].message.content.strip()

            # Làm sạch kết quả
            if translation.startswith("Vietnamese:"):
                translation = translation.replace("Vietnamese:", "").strip()

            return translation

        except Exception as e:
            if self.debug:
                print(f"✗ Lỗi dịch câu hỏi: {e}")
            return f"[Lỗi dịch] {question}"

    def augment_data(self, start_idx: int = 0, end_idx: int = None) -> List[Dict[str, Any]]:
        """Tạo dữ liệu tăng cường bằng few-shot translation"""
        # Tính số mẫu từ start và end
        if end_idx is None:
            actual_samples = 4000
            print(f"\n🔄 Đang tạo {actual_samples} mẫu dữ liệu tăng cường...")
            print(f"   📍 Bắt đầu từ: {start_idx}")
        else:
            actual_samples = end_idx - start_idx
            print(f"\n🔄 Đang tạo {actual_samples} mẫu dữ liệu tăng cường...")
            print(f"   📍 Phạm vi: từ {start_idx} đến {end_idx - 1}")

        # BƯỚC 1: Load tất cả câu hỏi đã tồn tại (manual + logs.txt)
        existing_questions = self.load_existing_questions()

        augmented_data = []
        used_train_indices = set()
        attempts = 0
        max_attempts = actual_samples * 10  # Tránh vòng lặp vô tận

        # Progress bar với tqdm
        with tqdm(total=actual_samples, desc="Translating questions", unit="samples") as pbar:
            while len(augmented_data) < actual_samples and attempts < max_attempts:
                attempts += 1

                # BƯỚC 2: Random chọn 1 mẫu từ train.json
                train_idx = random.randint(0, len(self.train_data) - 1)

                # Skip nếu index đã dùng rồi
                if train_idx in used_train_indices:
                    continue

                train_sample = self.train_data[train_idx]
                train_question = train_sample['question']

                # BƯỚC 3: Kiểm tra câu hỏi đã tồn tại chưa?
                if train_question in existing_questions:
                    used_train_indices.add(train_idx)
                    continue  # Random lại

                try:
                    # BƯỚC 4: Tìm similar examples bằng KNN và dịch
                    similar_examples = self.find_similar_examples(train_question, k=2)

                    # Dịch câu hỏi với few-shot
                    translation = self.translate_question(
                        question=train_question,
                        schema=train_sample['schema'],
                        examples=similar_examples
                    )

                    # Tạo mẫu dữ liệu mới
                    augmented_sample = {
                        'question': train_question,
                        'schema': train_sample['schema'],
                        'cypher': train_sample['cypher'],
                        'translation': translation
                    }

                    augmented_data.append(augmented_sample)
                    used_train_indices.add(train_idx)

                    # Thêm câu hỏi mới vào existing_questions để tránh trùng lặp
                    existing_questions.add(train_question)

                    pbar.update(1)

                    # Delay ngắn để tránh rate limit
                    time.sleep(0.2)

                except Exception as e:
                    if self.debug:
                        tqdm.write(f"✗ Lỗi mẫu {train_idx}: {str(e)[:50]}...")
                    used_train_indices.add(train_idx)
                    continue

        if attempts >= max_attempts:
            print(f"⚠️ Đạt giới hạn {max_attempts} lần thử, tạo được {len(augmented_data)}/{actual_samples} mẫu")

        print(f"✅ Hoàn thành: {len(augmented_data)} mẫu dữ liệu tăng cường")
        return augmented_data

    def save_checkpoint(self, data: List[Dict[str, Any]], checkpoint_file: str):
        """Lưu checkpoint"""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.debug:
                print(f"✗ Lỗi lưu checkpoint: {e}")

    def load_checkpoint(self, checkpoint_file: str) -> List[Dict[str, Any]]:
        """Load checkpoint"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Load checkpoint: {len(data)} mẫu từ {checkpoint_file}")
            return data
        except Exception as e:
            if self.debug:
                print(f"⚠️ Không load được checkpoint {checkpoint_file}: {e}")
            return []

    def save_final_data(self, augmented_data: List[Dict[str, Any]], output_file: str = '../data/augmented_data.json'):
        """Lưu CHỈ dữ liệu mới được dịch - KHÔNG bao gồm dữ liệu manual cũ"""
        print(f"\n💾 Đang lưu dữ liệu mới vào {output_file}...")

        # CHỈ lưu dữ liệu augmented_data (dữ liệu mới được dịch)
        # KHÔNG gộp với manual_data_clean như trước
        final_data = augmented_data

        # Lưu file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Đã lưu {len(final_data)} mẫu dữ liệu MỚI:")
        print(f"   📊 Dữ liệu tăng cường (mới dịch): {len(augmented_data)} mẫu")
        print(f"   📁 File: {output_file}")
        print(f"   ⚠️ Lưu ý: File chỉ chứa dữ liệu MỚI, không bao gồm manual_translation.json")

        return output_file

    def run_pipeline(self, start_idx: int = 0, end_idx: int = None, output_file: str = None):
        """Chạy pipeline đơn giản - không có checkpoint/resume"""
        print("🚀 Data Augmentation Pipeline - Few-Shot Translation")
        print("="*60)

        try:
            # Load dữ liệu
            if not self.load_data():
                return None

            # Tạo dữ liệu tăng cường
            augmented_data = self.augment_data(start_idx, end_idx)

            if not augmented_data:
                print("❌ Không tạo được dữ liệu tăng cường")
                return None

            # Tạo tên file output mặc định
            if output_file is None:
                if end_idx is None:
                    output_file = f'../data/augmented_data_{start_idx}_{start_idx + 4000}.json'
                else:
                    output_file = f'../data/augmented_data_{start_idx}_{end_idx}.json'

            # Lưu kết quả cuối
            result_file = self.save_final_data(augmented_data, output_file)

            print(f"\n🎉 Hoàn thành pipeline!")
            print(f"📁 Kết quả: {result_file}")

            return result_file

        except Exception as e:
            print(f"\n💥 Lỗi pipeline: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def load_existing_questions(self):
        """Load tất cả câu hỏi đã tồn tại từ manual_translation và logs.txt"""
        existing_questions = set()

        # Load từ manual_translation_with_embeddings.json
        for item in self.manual_translation_data:
            existing_questions.add(item['question'])

        # Load từ logs.txt nếu tồn tại
        logs_file = '../data/logs.txt'
        if os.path.exists(logs_file):
            try:
                with open(logs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        question = line.strip()
                        if question:
                            existing_questions.add(question)
                if self.debug:
                    print(f"✓ Load thêm câu hỏi từ {logs_file}")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Không đọc được {logs_file}: {e}")

        if self.debug:
            print(f"🔍 Tổng {len(existing_questions)} câu hỏi đã tồn tại (cần tránh)")

        return existing_questions

def main():
    parser = argparse.ArgumentParser(description='Data Augmentation Pipeline - Few-Shot Translation')
    parser.add_argument('--start', '-s',
                       type=int, default=0,
                       help='Vị trí bắt đầu (default: 0)')
    parser.add_argument('--end', '-e',
                       type=int, default=None,
                       help='Vị trí kết thúc (default: None, sẽ tạo 4000 mẫu)')
    parser.add_argument('--output', '-o',
                       type=str, default=None,
                       help='File output (default: ../data/augmented_data_{start}_{end}.json)')
    parser.add_argument('--debug',
                       action='store_true',
                       help='Bật chế độ debug')

    args = parser.parse_args()

    # Tính số mẫu từ start và end
    if args.end is None:
        num_samples = 4000
        end_display = f"{args.start + 4000}"
    else:
        num_samples = args.end - args.start
        end_display = str(args.end)

    print("🤖 Data Augmentation Pipeline")
    print("=" * 40)
    print(f"🎯 Samples: {num_samples}")
    print(f"📍 Range: {args.start} -> {end_display}")
    if args.output:
        print(f"📁 Output: {args.output}")
    else:
        output_name = f"../data/augmented_data_{args.start}_{end_display}.json"
        print(f"📁 Output: {output_name}")
    print("=" * 40)

    # Khởi tạo và chạy pipeline
    pipeline = DataAugmentationPipeline(debug=args.debug)
    result = pipeline.run_pipeline(
        start_idx=args.start,
        end_idx=args.end,
        output_file=args.output
    )

    if result:
        print(f"\n✅ Success! Result: {result}")
    else:
        print("\n❌ Pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main()
