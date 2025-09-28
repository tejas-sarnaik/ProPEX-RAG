import os
import json
import numpy as np
from tqdm import tqdm
import time

from rag_ppr_retriever import rank_passages_by_ppr, load_fact_triples_and_embeddings
from knowledge_graph_core import connect_to_memgraph, kg_builder
from config import (
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_EMBEDDING_VERSION
)
from openai import AzureOpenAI
from prompts.hotpot_prompt import build_hotpot_prompt

client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_EMBEDDING_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)

def generate_answer(question, passages, top_entities, top_triples):
    prompt = build_hotpot_prompt(question, passages, top_entities, top_triples)
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def write_final_debug_entry_strict_format(entry, file_handle):
    file_handle.write('{\n')
    for idx, (key, value) in enumerate(entry.items()):
        comma = ',' if idx < len(entry) - 1 else ''
        file_handle.write(f'  "{key}": ')
        if isinstance(value, list):
            if key == "query_entities" or key == "retrieved_passage_ids":
                file_handle.write(json.dumps(value, separators=(',', ': ')) + comma + '\n')
            else:
                file_handle.write('[\n')
                for i, item in enumerate(value):
                    item_comma = ',' if i < len(value) - 1 else ''
                    file_handle.write('    ' + json.dumps(item, separators=(',', ': ')) + item_comma + '\n')
                file_handle.write('  ]' + comma + '\n')
        else:
            file_handle.write(json.dumps(value, indent=2, separators=(',', ': ')) + comma + '\n')
    file_handle.write('}\n\n')

def run_qa_pipeline(input_file, output_file, top_k=5, max_examples=None):
    with open(input_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    context_lookup = {ex["_id"]: [" ".join(p[1]) for p in ex["context"]] for ex in qa_data}
    connect_to_memgraph()

    embedding_cache = os.path.join(kg_builder.data_dir, "entity_embeddings_cache_1_passage_data_with_ner_triples.json")
    if not hasattr(kg_builder, "entity_embeddings") or not kg_builder.entity_embeddings:
        if os.path.exists(embedding_cache):
            with open(embedding_cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if isinstance(cache, dict):
                    kg_builder.entity_embeddings = {k: np.array(v) for k, v in cache.items()}
                elif isinstance(cache, list):
                    kg_builder.entity_embeddings = {
                        item["entity"]: np.array(item["embedding"])
                        for item in cache if "entity" in item and "embedding" in item
                    }
            print(f"✅ Loaded {len(kg_builder.entity_embeddings)} entity embeddings.")
        else:
            print(f"❌ Entity embedding cache not found: {embedding_cache}")
            return

    if not getattr(kg_builder, "fact_triples", None) or not getattr(kg_builder, "fact_embeddings", None):
        load_fact_triples_and_embeddings()

    cursor = kg_builder.mg_conn.cursor()
    predictions = []

    final_debug_path = os.path.join(kg_builder.data_dir, "final_debug_trace", "final_debug_trace_sample_data.jsonl")
    os.makedirs(os.path.dirname(final_debug_path), exist_ok=True)

    with open(final_debug_path, "w", encoding="utf-8") as debug_file:
        for i, sample in enumerate(tqdm(qa_data[:max_examples], desc="Generating answers")):
            qid = sample["_id"]
            question = sample["question"]
            gold_answer = sample["answer"]

            try:
                result = rank_passages_by_ppr(
                    question, top_k=top_k, qid=qid, return_entities=True
                )
                if isinstance(result, tuple) and len(result) == 3:
                    retrievals, top_entities, top_triples = result
                else:
                    retrievals, top_entities = result
                    top_triples = []

                passage_ids = [r["passage_id"] for r in retrievals]

                contexts = []
                for pid in passage_ids:
                    cursor.execute("""
                        MATCH (p:Passage {passage_id: $pid})
                        RETURN coalesce(p.text, p.title) AS content
                    """, {"pid": pid})
                    row = cursor.fetchone()
                    if row and row[0]:
                        contexts.append(row[0].strip())
                    else:
                        print(f"⚠️ Passage {pid} not found in KG for {qid}")

                if not contexts:
                    fallback = context_lookup.get(qid, [])
                    if fallback:
                        print(f"🔁 Fallback: Using HotpotQA context for {qid}")
                        contexts = fallback

                predicted_answer = (
                    generate_answer(question, contexts, top_entities, top_triples)
                    if contexts else "ERROR: No context found"
                )

                predictions.append({
                    "id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "retrieved_passage_ids": passage_ids,
                    "retrieved_passages": contexts,
                    "top_triples": top_triples,
                    "top_entities": top_entities
                })

                debug_trace_path = os.path.join(kg_builder.data_dir, "debug_trace", f"{qid}.json")
                if os.path.exists(debug_trace_path):
                    with open(debug_trace_path, "r", encoding="utf-8") as trace_f:
                        debug_data = json.load(trace_f)
                else:
                    debug_data = {}

                debug_data.update({
                    "gold_answer": gold_answer,
                    "retrieved_passage_ids": passage_ids,
                    "retrieved_passages": contexts,
                    "predicted_answer": predicted_answer
                })

                write_final_debug_entry_strict_format(debug_data, debug_file)

            except Exception as e:
                print(f"[{qid}] ❌ Error: {e}")
                error_entry = {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "retrieved_passage_ids": [],
                    "retrieved_passages": [],
                    "predicted_answer": f"ERROR: {e}"
                }
                json.dump(error_entry, debug_file, indent=2, separators=(',', ': '))
                debug_file.write("\n\n")
                predictions.append(error_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in predictions:
            json.dump(ex, f)
            f.write("\n")

    print(f"\n✅ Saved {len(predictions)} predictions to: {output_file}")
    print(f"✅ Saved merged debug trace to: {final_debug_path}")

if __name__ == "__main__":
    input_file = "./datasets/2wikimultihopqa.json"
    output_file = "./final_output_dataset/testing_sample_dataset.jsonl"

    print("🚀 Starting QA Pipeline on HotpotQA...")
    run_qa_pipeline(
        input_file=input_file,
        output_file=output_file,
        top_k=5,
        max_examples=None
    )
    print("✅ QA Pipeline run complete!")