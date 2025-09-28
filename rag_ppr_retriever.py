####################################################################################################################################
#### The following approach is testing the pass the triples and passages in the prompt to answer generating process on 23/07/25 ####
### The final F1 score is come by using following approach is 76.84 testting on 1000QA with prompt change on 24/07/2025 ###
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from knowledge_graph_core import kg_builder, connect_to_memgraph
from config import (
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_EMBEDDING_VERSION
)
from prompts.triple_filter_prompt import build_fact_selection_prompt
from prompts.sampleqa_prompt import build_sampleqa_prompt
from openai import AzureOpenAI

# === Load fact triples and embeddings efficiently ===
FACT_FILE = "./output_directory/output_entity_facts_triplets/filtered_fact_triples_all.json"
FACT_EMB_FILE = "./output_directory/triple_embeddings.npy"

kg_builder.fact_triples = None
kg_builder.fact_embeddings = None

def load_fact_triples_and_embeddings():
    print("⚡ Loading fact triples and embeddings...")
    with open(FACT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        triples = data["all_triples"]
    else:
        triples = data
    kg_builder.fact_triples = triples
    kg_builder.fact_embeddings = np.load(FACT_EMB_FILE)
    assert len(kg_builder.fact_triples) == kg_builder.fact_embeddings.shape[0], (
        f"Triples and embeddings count mismatch: {len(kg_builder.fact_triples)} vs {kg_builder.fact_embeddings.shape[0]}"
    )
    print(f"✅ Loaded {len(kg_builder.fact_triples)} fact triples and {kg_builder.fact_embeddings.shape[0]} embeddings.")

# -----------------------------------------
ENTITY_PROP_WEIGHT = 1.0
SIMILARITY_PROP_WEIGHT = 0.6
RELATED_PROP_WEIGHT = 0.4
FACT_RERANK_WEIGHT = 0.7

def embed_question(question: str):
    try:
        response = kg_builder.client.embeddings.create(
            model=kg_builder.embedding_deployment,
            input=[question]
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def get_top_k_facts(question: str, k=10):
    q_embedding = embed_question(question)
    if q_embedding is None or kg_builder.fact_triples is None or kg_builder.fact_embeddings is None:
        print("❌ Fact triples or embeddings not loaded!")
        return []
    scores = cosine_similarity([q_embedding], kg_builder.fact_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [kg_builder.fact_triples[i] for i in top_indices]

def filter_triples_with_llm(question: str, triples: list, top_k: int = 4):
    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_EMBEDDING_VERSION,
        azure_endpoint=OPENAI_ENDPOINT
    )
    instruction = build_fact_selection_prompt(question, triples, top_k)

    print("\n🔵 Prompt sent to LLM:\n" + "=" * 40)
    print(instruction)
    print("=" * 40)

    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.0,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()

        print("\n🟢 Raw output received from LLM:\n" + "=" * 40)
        print(content)
        print("=" * 40)

        parsed = json.loads(content)
        return parsed.get("fact", [])
    except Exception as e:
        print(f"❌ LLM filtering error: {e}")
        return triples[:top_k]

def propagate_entity_scores(cursor, top_entities):
    print("\n🔁 Propagating entity scores in Memgraph...")
    cursor.execute("MATCH (n) REMOVE n.score")
    for name, score in top_entities:
        cursor.execute("""
            MATCH (e:Entity {name: $name})
            SET e.score = $score
        """, {"name": name, "score": float(score * ENTITY_PROP_WEIGHT)})
    cursor.execute("""
        MATCH (e1:Entity)-[:SIMILAR_TO]-(e2:Entity)
        WHERE e1.score IS NOT NULL
        WITH e2, sum(e1.score) AS propagated
        SET e2.score = coalesce(e2.score, 0) + $w * propagated
    """, {"w": SIMILARITY_PROP_WEIGHT})
    cursor.execute("""
        MATCH (e1:Entity)-[:RELATED]-(e2:Entity)
        WHERE e1.score IS NOT NULL
        WITH e2, sum(e1.score) AS propagated
        SET e2.score = coalesce(e2.score, 0) + $w * propagated
    """, {"w": RELATED_PROP_WEIGHT})
    cursor.execute("""
        MATCH (p:Passage)-[:MENTIONS]->(e:Entity)
        WHERE e.score IS NOT NULL
        WITH p, sum(e.score) AS p_score
        SET p.score = coalesce(p.score, 0) + p_score
    """)
    cursor.execute("""
        MATCH (e:Entity)-[:MENTIONS]->(p:Passage)
        WHERE e.score IS NOT NULL
        WITH p, sum(e.score) AS p_score
        SET p.score = coalesce(p.score, 0) + p_score
    """)
    print("\n🔍 Top scored entities after propagation:")
    cursor.execute("""
        MATCH (e:Entity)
        WHERE e.score IS NOT NULL
        RETURN e.name, e.score
        ORDER BY e.score DESC
        LIMIT 10
    """)
    for name, score in cursor.fetchall():
        print(f" - {name} (score: {score:.4f})")

def rerank_by_facts(question: str, passages: list):
    question_lower = question.lower()
    ranked = []
    for p in passages:
        cursor = kg_builder.mg_conn.cursor()
        cursor.execute("""
            MATCH (p:Passage {passage_id: $pid})-[:MENTIONS]-(e:Entity)
            RETURN e.name
        """, {"pid": p["passage_id"]})
        entities = [row[0].lower() for row in cursor.fetchall()]
        match_score = sum(1 for e in entities if e in question_lower)
        rerank_score = p["score"] + FACT_RERANK_WEIGHT * match_score
        ranked.append((p, rerank_score))
    return [x[0] for x in sorted(ranked, key=lambda x: x[1], reverse=True)]

DEBUG_TRACE_DIR = os.path.join(kg_builder.data_dir, "debug_trace_final")
os.makedirs(DEBUG_TRACE_DIR, exist_ok=True)

FINAL_TRACE_FILE = "./debug_trace_final/final_trace_all.jsonl"
os.makedirs(os.path.dirname(FINAL_TRACE_FILE), exist_ok=True)

def initialize_final_trace_file():
    with open(FINAL_TRACE_FILE, "w", encoding="utf-8") as f:
        pass  # Creates empty file to overwrite

def append_to_final_trace_strict_format(trace_entry):
    with open(FINAL_TRACE_FILE, "a", encoding="utf-8") as f:
        f.write('{\n')
        for idx, (key, value) in enumerate(trace_entry.items()):
            comma = ',' if idx < len(trace_entry) - 1 else ''
            f.write(f'  "{key}": ')
            if isinstance(value, list):
                if key == "query_entities":
                    # Inline array for query_entities
                    f.write(json.dumps(value, separators=(',', ': ')) + comma + '\n')
                else:
                    f.write('[\n')
                    for i, item in enumerate(value):
                        item_comma = ',' if i < len(value) - 1 else ''
                        f.write('    ' + json.dumps(item, separators=(',', ': ')) + item_comma + '\n')
                    f.write('  ]' + comma + '\n')
            else:
                f.write(json.dumps(value, indent=2, separators=(',', ': ')) + comma + '\n')
        f.write('}\n\n')

def rank_passages_by_ppr(question: str, top_k: int = 5, qid: str = "query1", return_entities: bool = False):
    print(f"\n🔍 [{qid}] Ranking passages using PPR + Synonymy + Fact Reranking")
    connect_to_memgraph()
    q_emb = embed_question(question)
    if q_emb is None:
        return ([], [], []) if return_entities else []

    # --- Fact Retrieval & LLM Filtering ---
    raw_facts = get_top_k_facts(question, k=10)
    filtered_facts = filter_triples_with_llm(question, raw_facts, top_k=4) if raw_facts else []

    # --- Seed Entity Creation ---
    if filtered_facts:
        seed_entities = {}
        for s, _, o in filtered_facts:
            for ent in [s.strip(), o.strip()]:
                seed_entities[ent] = seed_entities.get(ent, 0.0) + 1.0
        max_score = max(seed_entities.values(), default=1.0)
        top_entities = [(e, score / max_score) for e, score in seed_entities.items()]
    else:
        print("⚠️ No facts passed LLM filtering. Falling back to entity similarity.")
        top_entities = get_top_k_similar_entities(q_emb, k=10)

    # --- Propagation ---
    cursor = kg_builder.mg_conn.cursor()
    propagate_entity_scores(cursor, top_entities)
    cursor.execute("""
        MATCH (p:Passage)
        WHERE p.score IS NOT NULL
        RETURN p.passage_id AS id, p.title AS title, p.score AS score
        ORDER BY p.score DESC
        LIMIT $top_k
    """, {"top_k": max(30, top_k)})
    rows = cursor.fetchall()
    if not rows:
        print("❗ No passages retrieved after propagation. Falling back to top 500 passages by ID.")
        cursor.execute("""
            MATCH (p:Passage)
            RETURN p.passage_id AS id, p.title AS title, 0.0 AS score
            ORDER BY p.passage_id ASC
            LIMIT 500
        """)
        rows = cursor.fetchall()

    seed_names = set([name.lower() for name, _ in top_entities])
    passages = []
    for row in rows:
        bonus = 0.5 if any(e in row[1].lower() for e in seed_names) else 0.0
        passages.append({
            "question_id": qid,
            "passage_id": row[0],
            "title": row[1],
            "score": row[2] + bonus
        })

    # --- Reranking ---
    top_results = rerank_by_facts(question, passages)[:top_k]

    # --- Debugging Output ---
    debug_entry = {
        "question_id": qid,
        "question": question,
        "query_entities": [e[0] for e in get_top_k_similar_entities(q_emb, k=10)],
        "top_raw_facts": raw_facts,
        "filtered_facts": filtered_facts,
        "seed_entities": top_entities,
        "propagated_entities_top10": [],
        "top_30_passages_before_rerank": passages[:30],
        "final_top_5_passages": top_results
    }
    # Top 10 propagated entities
    cursor.execute("""
        MATCH (e:Entity)
        WHERE e.score IS NOT NULL
        RETURN e.name, e.score
        ORDER BY e.score DESC
        LIMIT 10
    """)
    debug_entry["propagated_entities_top10"] = [{"name": row[0], "score": row[1]} for row in cursor.fetchall()]

    # Save individual debug file
    per_question_debug_file = os.path.join(DEBUG_TRACE_DIR, f"{qid}.json")
    with open(per_question_debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_entry, f, indent=2, separators=(',', ': '))

    # Append to final_trace_all.json
    append_to_final_trace_strict_format(debug_entry)

    # Save retrieved passages jsonl
    out_dir = os.path.join(kg_builder.data_dir, "retrievals")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{qid}_ppr.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in top_results:
            json.dump(item, f)
            f.write("\n")
    print(f"💾 Results saved to: {out_path}")

    if return_entities:
        return top_results, [e[0] for e in top_entities], filtered_facts

    if not top_results:
        print("🚨 Still no top passages retrieved after reranking.")
        print("📄 Available passages fetched (pre-rerank):")
        for p in passages[:20]:
            print(f"   - [{p['passage_id']}] {p['title']} (Score: {p['score']:.4f})")

    return top_results

def get_top_k_similar_entities(question_embedding: np.ndarray, k=10):
    if not kg_builder.entity_embeddings:
        print("❌ No entity embeddings found.")
        return []
    names, vectors = zip(*kg_builder.entity_embeddings.items())
    sim = cosine_similarity([question_embedding], vectors)[0]
    top_indices = np.argsort(sim)[::-1][:k]
    return [(names[i], sim[i]) for i in top_indices]

# ========== MAIN BLOCK ==========
if __name__ == "__main__":
    embedding_cache = "./output_directory/entity_embeddings_cache_1_passage_data_with_ner_triples.json"
    connect_to_memgraph()

    # === ENTITY EMBEDDINGS ===
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
        print(f"❌ Embedding cache not found at {embedding_cache}")
        exit()

    # === PATCHED: LOAD FACT TRIPLES AND EMBEDDINGS FAST ===
    load_fact_triples_and_embeddings()

    # === QUICK TESTS ===
    test_questions = [
        "Which of the following artists is associated with the band One eskimO?",
        "What is the Limestone Coast known for besides being a government region?",
        "Which musician has been a part of more than one heavy metal or thrash band?",
        "What was the population of Marufabad according to the 2006 census?"
    ]
    for i, q in enumerate(test_questions):
        print(f"\n===== TEST CASE {i+1}: {q}")
        passages = rank_passages_by_ppr(q, top_k=5, qid=f"debug_test_{i+1}", return_entities=False)
