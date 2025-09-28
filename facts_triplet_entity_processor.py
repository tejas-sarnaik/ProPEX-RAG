import json
import re
import time
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Set
from tqdm import tqdm
import logging
import pickle
from pathlib import Path
import gc
import psutil
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_EMBEDDING_DEPLOYMENT,
    OPENAI_EMBEDDING_VERSION,
    SIMILARITY_THRESHOLD,
    EMBEDDING_BATCH_SIZE
)
from prompts.prompts import ENTITY_EXTRACTION_PROMPT, TRIPLE_EXTRACTION_PROMPT

# Optional: import other LLM or embedding clients here (e.g., llama)

import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripletEntityProcessor:
    def __init__(self, input_file_path: str, output_dir, batch_size: int = 100):
        self.input_file_path = input_file_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.passage_file = self.output_dir / "1_passage_data_with_ner_triples.json"
        self.triples_file = self.output_dir / "filtered_fact_triples_all.json"

        self.processed_count = 0
        self.total_passages = 0
        self.unique_entities: Set[str] = set()
        self.total_triples = 0
        self.relation_counts = {}
        self.entity_embeddings = {}

        # Flexible client initialization
        self.llm_provider = "openai"
        self.client = self._init_client()

        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.embedding_batch_size = EMBEDDING_BATCH_SIZE

        self.load_checkpoint()

    def _init_client(self):
        if self.llm_provider == "openai":
            return openai.AzureOpenAI(
                api_key=OPENAI_API_KEY,
                api_version=OPENAI_EMBEDDING_VERSION,
                azure_endpoint=OPENAI_ENDPOINT
            )
        elif self.llm_provider == "llama":
            # Placeholder for future integration with LLaMA model (e.g., via vLLM or HuggingFace)
            raise NotImplementedError("LLaMA integration not yet implemented.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def save_checkpoint(self):
        checkpoint_data = {
            "processed_count": self.processed_count,
            "total_passages": self.total_passages,
            "total_triples": self.total_triples,
            "unique_entities_count": len(self.unique_entities),
            "relation_counts": self.relation_counts
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                self.processed_count = checkpoint_data.get("processed_count", 0)
                self.total_passages = checkpoint_data.get("total_passages", 0)
                self.total_triples = checkpoint_data.get("total_triples", 0)
                self.relation_counts = checkpoint_data.get("relation_counts", {})
                print(f"📍 Resumed from checkpoint: {self.processed_count} passages processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self.processed_count = 0

    def normalize_entity(self, entity: str) -> str:
        if not entity:
            return ""
        return entity.strip().replace('\n', ' ').replace('\t', ' ')

    def extract_named_entities(self, passage_text: str) -> List[str]:
        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": passage_text}
        ]
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            response_text = response.choices[0].message.content
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                entities = json.loads(match.group(0)).get("named_entities", [])
                return [self.normalize_entity(e) for e in entities if e]
            return []
        except Exception as e:
            logger.warning(f"NER extraction error: {e}")
            return []

    def extract_rdf_triples(self, passage_text: str, named_entities: List[str]) -> List[List[str]]:
        named_entity_json = json.dumps({"named_entities": named_entities})
        current_input = f"Paragraph:\n```\n{passage_text}\n```\n{named_entity_json}"
        messages = [
            {"role": "system", "content": TRIPLE_EXTRACTION_PROMPT},
            {"role": "user", "content": current_input}
        ]
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            response_text = response.choices[0].message.content
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                triples = json.loads(match.group(0)).get("triples", [])
                return [
                    [self.normalize_entity(x) for x in t]
                    for t in triples if isinstance(t, list) and len(t) == 3
                ]
            return []
        except Exception as e:
            logger.warning(f"Triple extraction error: {e}")
            return []

    def load_data(self):
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            processed = []
            for i, d in enumerate(data):
                processed.append({
                    "passage_id": i + 1,
                    "title": d.get("title", f"passage_{i+1}"),
                    "text": d.get("text", "")
                })
            self.total_passages = len(processed)
            return processed
        except Exception as e:
            logger.error(f"Data load error: {e}")
            return []

    def append_passage(self, passage_data: Dict, is_last: bool = False):
        with open(self.passage_file, 'a', encoding='utf-8') as f:
            if self.processed_count > 0:
                f.write(',\n')
            json.dump(passage_data, f, indent=2, ensure_ascii=False)
            if is_last:
                f.write('\n]')

    def update_triples_file(self, passage_id: int, triples: List[List[str]]):
        try:
            if not self.triples_file.exists():
                with open(self.triples_file, 'w', encoding='utf-8') as f:
                    json.dump({"triples_by_passage": {}, "all_triples": []}, f, indent=2)
            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["triples_by_passage"][str(passage_id)] = triples
            for t in triples:
                data["all_triples"].append({"passage_id": passage_id, "triple": t})
            with open(self.triples_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Triple file update error: {e}")

    def run_complete_processing(self):
        data = self.load_data()
        if not data:
            return
        if not self.passage_file.exists() or self.processed_count == 0:
            with open(self.passage_file, 'w', encoding='utf-8') as f:
                f.write('[\n')

        remaining = len(data) - self.processed_count

        # tqdm: wrap range, leave=True so last bar persists, dynamic_ncols for resizing, ascii=True for more compatibility
        with tqdm(
            total=remaining,
            initial=0,
            desc="Processing passages",
            unit="passage",
            dynamic_ncols=True,
            ascii=True,
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:

            for i in range(self.processed_count, len(data)):
                item = data[i]
                passage_id = item["passage_id"]
                title = item.get("title", f"passage_{passage_id}")
                text = item["text"]

                pbar.set_description(f"Passage {passage_id}: {title[:40]}...")

                # Step 1: Extract Entities
                entities = self.extract_named_entities(text)
                # Optionally show in tqdm bar (comment out if too verbose)
                pbar.set_postfix({"entities": len(entities)})

                # Step 2: Extract Triples
                triples = self.extract_rdf_triples(text, entities)
                pbar.set_postfix({"triples": len(triples)})

                # Update tracking
                self.unique_entities.update(entities)
                self.total_triples += len(triples)
                for t in triples:
                    if len(t) == 3:
                        r = t[1]
                        self.relation_counts[r] = self.relation_counts.get(r, 0) + 1

                # Save passage result
                passage_data = {
                    **item,
                    "extracted_entities": {"entities": entities, "entity_count": len(entities)},
                    "extracted_triples": {"triples": triples, "triple_count": len(triples)}
                }

                self.append_passage(passage_data, is_last=(i == len(data) - 1))
                self.update_triples_file(passage_id, triples)
                self.processed_count += 1

                if self.processed_count % self.batch_size == 0:
                    self.save_checkpoint()
                    gc.collect()
                pbar.update(1)
                # Optional: Show memory usage in postfix
                pbar.set_postfix({"mem (MB)": round(self.get_memory_usage(), 2)})

        self.save_checkpoint()
        print("✅ Extraction complete")
        print(f"Total passages processed: {self.processed_count}")
        print(f"Total unique entities: {len(self.unique_entities)}")
        print(f"Total triples extracted: {self.total_triples}")