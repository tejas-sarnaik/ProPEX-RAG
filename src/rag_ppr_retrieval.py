"""
===============================================================================
This module implements the retrieval stage for ProPEX-RAG in a clear and well-documented manner. The retrieval stage accepts a natural language question and returns a ranked list of candidate passages from the knowledge graph.
===============================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_graph_core import kg_builder, connect_to_memgraph
from prompts.triple_filter_prompt import build_fact_selection_prompt
from prompts.sampleqa_prompt import build_sampleqa_prompt  # kept for future use

from config import (
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_EMBEDDING_VERSION,
)

# The OpenAI Azure client is imported here so that this module can perform
# optional language model calls for fact filtering. If you swap providers,
# change client creation in _get_llm_client.
from openai import AzureOpenAI


# =============================================================================
# Configuration section
# =============================================================================

# Paths for fact triples and their embeddings. These should be produced during
# the offline indexing or graph construction stage.
FACT_FILE = "./output_directory/output_entity_facts_triplets/filtered_fact_triples_all.json"
FACT_EMB_FILE = "./output_directory/triple_embeddings.npy"

# Output and debug directories. The builder provides a data directory; we use
# it as the root to keep artifacts organized with the rest of the pipeline.
DEBUG_TRACE_DIR = os.path.join(getattr(kg_builder, "data_dir", "./output_directory"), "debug_trace_final")
FINAL_TRACE_FILE = os.path.join(DEBUG_TRACE_DIR, "final_trace_all.jsonl")
RETRIEVALS_DIR = os.path.join(getattr(kg_builder, "data_dir", "./output_directory"), "retrievals")

# Weights for graph propagation and reranking policies. Adjust with care and
# record changes for reproducibility.
ENTITY_PROP_WEIGHT = 1.0
SIMILARITY_PROP_WEIGHT = 0.6
RELATED_PROP_WEIGHT = 0.4
FACT_RERANK_WEIGHT = 0.7
TITLE_BONUS_WEIGHT = 0.5

# Safety caps to avoid returning too few or too many passages during the
# harvesting stage before reranking.
MIN_HARVEST = 30
MAX_FALLBACK_HARVEST = 500

# Default pool sizes for facts and filtered facts.
DEFAULT_RAW_FACT_K = 10
DEFAULT_FILTERED_FACT_K = 4


# =============================================================================
# In-memory caches owned by the builder (populated on demand)
# =============================================================================

# We attach these caches to the builder to share state with other modules while
# keeping a single source of truth for loaded data.
kg_builder.fact_triples = getattr(kg_builder, "fact_triples", None)
kg_builder.fact_embeddings = getattr(kg_builder, "fact_embeddings", None)


# =============================================================================
# Utility and support functions
# =============================================================================

def _ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(DEBUG_TRACE_DIR, exist_ok=True)
    os.makedirs(RETRIEVALS_DIR, exist_ok=True)


def _get_llm_client() -> AzureOpenAI:
    """
    Return an LLM client instance based on configuration.

    In the default configuration this returns an Azure OpenAI client. If you
    change providers, update this function to construct and return the client
    for your target provider.
    """
    return AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_EMBEDDING_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
    )


def _safe_json_dump(obj: dict, fp, indent: int = 2) -> None:
    """
    Dump a dictionary to a file handle as JSON. This wrapper keeps a single
    place to adjust serialization options if needed later.
    """
    json.dump(obj, fp, indent=indent, separators=(",", ": "))


def _strict_append_jsonl(path: str, entry: dict) -> None:
    """
    Append a dictionary as a single JSON object to a JSONL file. The file is
    created if it does not exist.
    """
    with open(path, "a", encoding="utf-8") as f:
        _safe_json_dump(entry, f, indent=2)
        f.write("\n")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_maybe_list(path: str) -> Sequence:
    """
    Read JSON that may be a list or dictionary. Return the parsed object for
    flexible handling in the caller.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _np_load(path: str) -> np.ndarray:
    return np.load(path)


def _as_np(x) -> np.ndarray:
    return np.array(x)


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a vector to unit range. If all values are equal, return zeros.
    """
    vmin = float(values.min()) if values.size else 0.0
    vmax = float(values.max()) if values.size else 0.0
    rng = max(vmax - vmin, eps)
    return (values - vmin) / rng if values.size else values


def _sorted_top_k(values: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top k values in descending order.
    """
    if values.size == 0:
        return np.array([], dtype=int)
    k = max(0, min(k, values.size))
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


# =============================================================================
# Loading fact triples and embeddings
# =============================================================================

def load_fact_triples_and_embeddings() -> None:
    """
    Load fact triples and their embeddings into the builder caches.

    The triples file may be a list or it may be a dictionary with a specific
    key that contains a list. The function supports both layouts. The function
    validates that the number of triples matches the number of embedding rows.
    """
    data = _read_json_maybe_list(FACT_FILE)
    if isinstance(data, dict):
        triples = data.get("all_triples", [])
    else:
        triples = data

    embeddings = _np_load(FACT_EMB_FILE)

    if len(triples) != embeddings.shape[0]:
        raise ValueError(
            f"Triples and embeddings count mismatch: {len(triples)} vs {embeddings.shape[0]}"
        )

    kg_builder.fact_triples = triples
    kg_builder.fact_embeddings = embeddings

    print(f"Loaded {len(triples)} fact triples and {embeddings.shape[0]} embeddings.")


# =============================================================================
# Embedding the question and entity similarity utilities
# =============================================================================

def embed_question(question: str) -> Optional[np.ndarray]:
    """
    Compute an embedding for a question string using the configured embedding
    deployment. Return None if embedding fails.
    """
    try:
        response = kg_builder.client.embeddings.create(
            model=kg_builder.embedding_deployment,
            input=[question],
        )
        emb = response.data[0].embedding
        return _as_np(emb)
    except Exception as exc:
        print(f"Embedding error: {exc}")
        return None


def get_top_k_similar_entities(question_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """
    Compute cosine similarity between the question embedding and each entity
    embedding in the cache. Return the top k entities with their similarity
    scores. If no entity embeddings are present, return an empty list.
    """
    if not getattr(kg_builder, "entity_embeddings", None):
        print("No entity embeddings found in builder cache.")
        return []

    names, vectors = zip(*kg_builder.entity_embeddings.items())
    matrix = np.vstack(vectors)
    sim = cosine_similarity([question_embedding], matrix)[0]
    top_idx = _sorted_top_k(sim, k)
    return [(names[i], float(sim[i])) for i in top_idx]


# =============================================================================
# Fact retrieval and optional LLM filtering
# =============================================================================

def get_top_k_facts(question: str, k: int = DEFAULT_RAW_FACT_K) -> List:
    """
    Retrieve a pool of top k fact triples by question-to-fact similarity.

    The function embeds the question and compares it with the fact embedding
    matrix. It returns a list of triples selected by highest similarity.
    """
    qemb = embed_question(question)
    if qemb is None:
        return []
    if kg_builder.fact_triples is None or kg_builder.fact_embeddings is None:
        print("Fact triples or embeddings are not loaded. Call load_fact_triples_and_embeddings first.")
        return []

    scores = cosine_similarity([qemb], kg_builder.fact_embeddings)[0]
    top_idx = _sorted_top_k(scores, k)
    return [kg_builder.fact_triples[i] for i in top_idx]


def filter_triples_with_llm(
    question: str,
    triples: List,
    top_k: int = DEFAULT_FILTERED_FACT_K,
) -> List:
    """
    Use a language model to filter an input list of triples.

    The function constructs an instruction using a builder for the triple
    selection prompt, calls the LLM, and parses the returned JSON. If the
    LLM call fails or if the output cannot be parsed, the function returns
    a fallback slice of the input list.
    """
    client = _get_llm_client()
    instruction = build_fact_selection_prompt(question, triples, top_k)

    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed.get("fact", []) or triples[:top_k]
    except Exception as exc:
        print(f"LLM filtering error: {exc}")
        return triples[:top_k]


# =============================================================================
# Score propagation and passage projection in Memgraph
# =============================================================================

def _clear_all_scores(cursor) -> None:
    """Remove score properties from all nodes before propagation begins."""
    cursor.execute("MATCH (n) REMOVE n.score")


def _seed_entity_scores(cursor, seeds: List[Tuple[str, float]], weight: float) -> None:
    """
    Seed entity nodes with scores. Each seed is a tuple of entity name and
    initial score. The weight scales the initial score.
    """
    for name, base in seeds:
        cursor.execute(
            """
            MATCH (e:Entity {name: $name})
            SET e.score = $score
            """,
            {"name": name, "score": float(base * weight)},
        )


def _propagate_over_similarity(cursor, weight: float) -> None:
    """
    Propagate scores over similarity relations. The operation reads scores
    from neighboring entities and adds a scaled contribution to each target.
    """
    cursor.execute(
        """
        MATCH (a:Entity)-[:SIMILAR_TO]-(b:Entity)
        WHERE a.score IS NOT NULL
        WITH b, sum(a.score) AS inc
        SET b.score = coalesce(b.score, 0) + $w * inc
        """,
        {"w": weight},
    )


def _propagate_over_related(cursor, weight: float) -> None:
    """
    Propagate scores over related relations using the same logic as the
    similarity step, but with a separate weight so the two effects can be
    tuned independently.
    """
    cursor.execute(
        """
        MATCH (a:Entity)-[:RELATED]-(b:Entity)
        WHERE a.score IS NOT NULL
        WITH b, sum(a.score) AS inc
        SET b.score = coalesce(b.score, 0) + $w * inc
        """,
        {"w": weight},
    )


def _project_entity_scores_to_passages(cursor) -> None:
    """
    Project entity scores into passage scores. The function traverses mention
    relations in both directions to account for either edge orientation.
    """
    cursor.execute(
        """
        MATCH (p:Passage)-[:MENTIONS]->(e:Entity)
        WHERE e.score IS NOT NULL
        WITH p, sum(e.score) AS s
        SET p.score = coalesce(p.score, 0) + s
        """
    )
    cursor.execute(
        """
        MATCH (e:Entity)-[:MENTIONS]->(p:Passage)
        WHERE e.score IS NOT NULL
        WITH p, sum(e.score) AS s
        SET p.score = coalesce(p.score, 0) + s
        """
    )


def propagate_entity_scores(cursor, top_entities: List[Tuple[str, float]]) -> None:
    """
    Orchestrate the propagation process. Clear existing scores, seed entities,
    propagate through similarity and related relations, then project to
    passages.
    """
    _clear_all_scores(cursor)
    _seed_entity_scores(cursor, top_entities, ENTITY_PROP_WEIGHT)
    _propagate_over_similarity(cursor, SIMILARITY_PROP_WEIGHT)
    _propagate_over_related(cursor, RELATED_PROP_WEIGHT)
    _project_entity_scores_to_passages(cursor)


def _inspect_top_entities(cursor, limit: int = 10) -> List[Tuple[str, float]]:
    """
    Return a sample of the highest scoring entities after propagation. The
    caller can use this for debug output.
    """
    cursor.execute(
        """
        MATCH (e:Entity)
        WHERE e.score IS NOT NULL
        RETURN e.name, e.score
        ORDER BY e.score DESC
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    rows = cursor.fetchall()
    return [(row[0], float(row[1])) for row in rows]


# =============================================================================
# Passage harvesting and reranking
# =============================================================================

def _harvest_passages_after_propagation(cursor, limit: int) -> List[Dict]:
    """
    Collect a set of candidate passages based on the score produced during
    projection. Result rows include passage id, passage title, and the score.
    """
    cursor.execute(
        """
        MATCH (p:Passage)
        WHERE p.score IS NOT NULL
        RETURN p.passage_id AS id, p.title AS title, p.score AS score
        ORDER BY p.score DESC
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    rows = cursor.fetchall()
    return [
        {"passage_id": row[0], "title": row[1], "score": float(row[2])}
        for row in rows
    ]


def _fallback_harvest_by_id(cursor, limit: int) -> List[Dict]:
    """
    If the graph does not yield scored passages, fall back to an ordered list
    of passages by id. This avoids empty results and makes failure cases
    inspectable.
    """
    cursor.execute(
        """
        MATCH (p:Passage)
        RETURN p.passage_id AS id, p.title AS title, 0.0 AS score
        ORDER BY p.passage_id ASC
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    rows = cursor.fetchall()
    return [
        {"passage_id": row[0], "title": row[1], "score": float(row[2])}
        for row in rows
    ]


def _title_bonus(pass_title: Optional[str], seed_names_lower: set) -> float:
    """
    Compute a simple title bonus when any seed entity name appears in the
    passage title. The function uses case-insensitive overlap.
    """
    if not pass_title:
        return 0.0
    t = pass_title.lower()
    return TITLE_BONUS_WEIGHT if any(n in t for n in seed_names_lower) else 0.0


def rerank_by_facts(question: str, passages: List[Dict]) -> List[Dict]:
    """
    Adjust passage order by a question overlap signal. The method queries the
    graph for entities that each passage mentions. It gives a small additive
    score for every entity name substring that appears in the lowercased
    question. This is a lightweight step that favors passages whose entities
    more directly match the question.
    """
    q = question.lower()
    ranked: List[Tuple[Dict, float]] = []
    cursor = kg_builder.mg_conn.cursor()
    for p in passages:
        cursor.execute(
            """
            MATCH (p:Passage {passage_id: $pid})-[:MENTIONS]-(e:Entity)
            RETURN e.name
            """,
            {"pid": p["passage_id"]},
        )
        names = [row[0] for row in cursor.fetchall()]
        overlap = sum(1 for name in names if name and name.lower() in q)
        score = float(p["score"]) + FACT_RERANK_WEIGHT * float(overlap)
        ranked.append((p, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked]


# =============================================================================
# Debug trace helpers
# =============================================================================

def initialize_final_trace_file() -> None:
    """Create or truncate the combined debug trace file."""
    _ensure_dirs()
    with open(FINAL_TRACE_FILE, "w", encoding="utf-8"):
        pass


def _collect_debug_snapshot(
    question_id: str,
    question: str,
    query_entities_preview: List[str],
    raw_facts: List,
    filtered_facts: List,
    seeds: List[Tuple[str, float]],
    propagated_top_entities: List[Tuple[str, float]],
    pre_rerank_passages: List[Dict],
    final_passages: List[Dict],
) -> Dict:
    """
    Build a dictionary that captures the retrieval process for a single query.
    This snapshot can be written to the combined JSONL file and to the per
    question file. It is designed to be comprehensive but still readable.
    """
    return {
        "question_id": question_id,
        "question": question,
        "query_entities": query_entities_preview,
        "top_raw_facts": raw_facts,
        "filtered_facts": filtered_facts,
        "seed_entities": [{"name": n, "weight": float(w)} for n, w in seeds],
        "propagated_entities_top": [
            {"name": n, "score": float(s)} for n, s in propagated_top_entities
        ],
        "top_passages_before_rerank": pre_rerank_passages,
        "final_top_passages": final_passages,
    }


def _write_debug_files(question_id: str, snapshot: Dict) -> None:
    """
    Write two outputs for each query. The function writes a per-query JSON
    file for focused debugging and appends the same snapshot to the combined
    JSONL file for global analysis.
    """
    _ensure_dirs()
    per_query_path = os.path.join(DEBUG_TRACE_DIR, f"{question_id}.json")
    with open(per_query_path, "w", encoding="utf-8") as f:
        _safe_json_dump(snapshot, f, indent=2)
    _strict_append_jsonl(FINAL_TRACE_FILE, snapshot)


# =============================================================================
# Main retrieval entry point
# =============================================================================

def rank_passages_by_ppr(
    question: str,
    top_k: int = 5,
    qid: str = "query1",
    return_entities: bool = False,
) -> List[Dict] | Tuple[List[Dict], List[str], List]:
    """
    Rank passages for a given question using a propagation-based approach
    over the knowledge graph, followed by a lightweight reranking step.

    Arguments
    ---------
    question:
        The input question in natural language.
    top_k:
        The number of final passages to return after reranking.
    qid:
        A unique identifier for the question. Used in filenames for debug
        outputs and retrieval artifacts.
    return_entities:
        If True, return a tuple that includes the final list of passages,
        the list of seed entities by name, and the list of filtered facts.

    Returns
    -------
    Either a list of passages or a tuple as described above. Each passage
    is a dictionary that includes the passage id, the title, and the score.
    """
    connect_to_memgraph()
    _ensure_dirs()

    # Step A. Embed the question. If this fails, return a safe empty result.
    q_emb = embed_question(question)
    if q_emb is None:
        if return_entities:
            return [], [], []
        return []

    # Step B. Retrieve a small pool of candidate facts by similarity.
    raw_facts = get_top_k_facts(question, k=DEFAULT_RAW_FACT_K)

    # Step C. Optionally filter the facts with a language model.
    filtered_facts = filter_triples_with_llm(question, raw_facts, top_k=DEFAULT_FILTERED_FACT_K) if raw_facts else []

    # Step D. Build seed entities from the filtered fact pool. The seeds
    # collect subjects and objects from the fact triples and assign counts.
    # The counts are normalized so the maximum receives weight one.
    if filtered_facts:
        freq: Dict[str, float] = {}
        for triple in filtered_facts:
            # Triple layout is expected to be [subject, relation, object] or a similar
            # three-element structure. We extract the subject and object strings.
            try:
                s, _, o = triple
            except Exception:
                # If the triple is not a simple three-tuple, skip with caution.
                continue
            for ent in (str(s).strip(), str(o).strip()):
                if not ent:
                    continue
                freq[ent] = freq.get(ent, 0.0) + 1.0
        if freq:
            max_count = max(freq.values())
            seeds = [(name, val / max_count if max_count > 0 else 0.0) for name, val in freq.items()]
        else:
            seeds = []
    else:
        # When filtering yields no facts, fall back to entity similarity seeds.
        candidates = get_top_k_similar_entities(q_emb, k=DEFAULT_RAW_FACT_K)
        seeds = [(name, float(score)) for name, score in candidates]

    # Step E. Propagate entity scores across the graph and project to passages.
    cursor = kg_builder.mg_conn.cursor()
    propagate_entity_scores(cursor, seeds)

    # Step F. Inspect a sample of high scoring entities for debug purposes.
    propagated_top = _inspect_top_entities(cursor, limit=10)

    # Step G. Harvest a candidate list of passages after propagation. Enforce a
    # minimum harvest so the reranker has material to work with.
    harvest_n = max(MIN_HARVEST, int(top_k))
    pre = _harvest_passages_after_propagation(cursor, limit=harvest_n)
    if not pre:
        # If no passages received scores, fall back to a neutral harvest by id.
        pre = _fallback_harvest_by_id(cursor, limit=MAX_FALLBACK_HARVEST)

    # Step H. Apply a small title bonus using seed names for additional recall.
    seed_names_lower = {n.lower() for n, _ in seeds}
    for item in pre:
        item["score"] = float(item.get("score", 0.0)) + _title_bonus(item.get("title"), seed_names_lower)

    # Step I. Rerank with a lightweight fact overlap signal and cut to top k.
    reranked = rerank_by_facts(question, pre)
    final = reranked[: int(top_k)]

    # Step J. Build and write debug snapshots.
    query_entities_preview = [name for name, _ in get_top_k_similar_entities(q_emb, k=10)]
    snapshot = _collect_debug_snapshot(
        question_id=qid,
        question=question,
        query_entities_preview=query_entities_preview,
        raw_facts=raw_facts,
        filtered_facts=filtered_facts,
        seeds=seeds,
        propagated_top_entities=propagated_top,
        pre_rerank_passages=pre[:30],  # keep the snapshot concise
        final_passages=final,
    )
    _write_debug_files(qid, snapshot)

    # Step K. Write the retrieval results as JSONL for external inspection.
    out_path = os.path.join(RETRIEVALS_DIR, f"{qid}_ppr.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in final:
            _safe_json_dump(row, f, indent=2)
            f.write("\n")

    # Step L. Return the chosen format.
    if return_entities:
        return final, [n for n, _ in seeds], filtered_facts
    return final


# =============================================================================
# Command line and quick tests
# =============================================================================

def _load_entity_embeddings_from_cache(cache_path: str) -> None:
    """
    Load entity embeddings into the builder cache from a JSON cache file that
    stores either a dictionary mapping names to vectors or a list of objects
    each containing an entity name and an embedding vector.
    """
    if not os.path.exists(cache_path):
        print(f"Embedding cache not found at {cache_path}")
        return

    data = _read_json(cache_path)
    result: Dict[str, np.ndarray] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            result[str(k)] = _as_np(v)
    elif isinstance(data, list):
        for item in data:
            name = item.get("entity")
            vec = item.get("embedding")
            if name is None or vec is None:
                continue
            result[str(name)] = _as_np(vec)
    else:
        print("Unexpected format for entity embeddings cache. Expected dict or list.")

    kg_builder.entity_embeddings = result
    print(f"Loaded {len(kg_builder.entity_embeddings)} entity embeddings.")


def _load_facts_and_embeddings() -> None:
    """
    Load fact triples and their embeddings, and report the sizes. This function
    provides a single call site for tests and manual runs.
    """
    load_fact_triples_and_embeddings()


def _demo_queries() -> List[str]:
    """
    Provide a small set of demonstration questions that exercise different
    parts of the pipeline. These are meant for sanity checks and do not
    replace a formal evaluation script.
    """
    return [
        "Which of the following artists is associated with the band One eskimO?",
        "What is the Limestone Coast known for besides being a government region?",
        "Which musician has been a part of more than one heavy metal or thrash band?",
        "What was the population of Marufabad according to the 2006 census?",
        "When did Maradona sign with Barcelona?",
    ]


def _run_demo(top_k: int = 5) -> None:
    """
    Run the retrieval pipeline on a fixed list of demonstration questions and
    print short summaries to standard output. This helps verify that the graph
    connection works and that the retrieval logic returns nonempty results.
    """
    _ensure_dirs()
    initialize_final_trace_file()
    connect_to_memgraph()

    # Try to find a default cache path under the builder directory. If this path
    # is not correct in your setup, update it here or pass it as a parameter.
    default_cache = os.path.join(
        getattr(kg_builder, "data_dir", "./output_directory"),
        "entity_embeddings_cache_1_passage_data_with_ner_triples.json",
    )
    _load_entity_embeddings_from_cache(default_cache)
    _load_facts_and_embeddings()

    qs = _demo_queries()
    for i, q in enumerate(qs, start=1):
        qid = f"debug_test_{i}"
        print(f"[{qid}] Question: {q}")
        t0 = time.time()
        results = rank_passages_by_ppr(q, top_k=top_k, qid=qid, return_entities=False)
        dt = time.time() - t0
        if results:
            head = results[0]
            print(f"[{qid}] Top passage id: {head.get('passage_id')}, title: {head.get('title')}, score: {head.get('score'):.4f}")
        else:
            print(f"[{qid}] No passages returned.")
        print(f"[{qid}] Retrieval time seconds: {dt:.2f}")
        print("-" * 60)


if __name__ == "__main__":
    # This main section provides a simple manual test harness. It connects to
    # the graph, loads caches, and runs a set of demonstration questions. For
    # full evaluation on a dataset, use the qa_pipeline driver.
    try:
        _run_demo(top_k=5)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Unhandled error in demo run: {exc}")
        sys.exit(1)