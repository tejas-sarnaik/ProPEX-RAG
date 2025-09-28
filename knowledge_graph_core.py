## New Approach 25_06_2025 Knowlage_graph_core.py ###
# Updated Knowledge Graph Builder Functions


import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import mgclient
import gc
from tqdm import tqdm
from typing import List, Dict
import openai
import os
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
from config import (
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_EMBEDDING_DEPLOYMENT,
    SIMILARITY_THRESHOLD,
    EMBEDDING_BATCH_SIZE,
    MEMGRAPH_HOST,
    MEMGRAPH_PORT,
    MEMGRAPH_USERNAME,
    MEMGRAPH_PASSWORD
)

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix
import warnings
warnings.filterwarnings('ignore')

# Check Memgraph availability
try:
    MEMGRAPH_AVAILABLE = True
    print("✅ Memgraph client available")
except ImportError:
    MEMGRAPH_AVAILABLE = False
    print("⚠️ Memgraph client not available. Install with: pip install pymgclient")

# ✅ Batching utility
def batched(iterable, batch_size):
    """Yield successive batch-sized chunks from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class KnowledgeGraphBuilder:
    def __init__(self, data_dir: str = "./output_directory"):
        self.data_dir = data_dir
        self.entities_data = None
        self.triples_data = None
        self.passages_data = None
        self.similarity_matrix = None
        self.entity_embeddings = {}
        self.valid_entities = []
        self.ready_to_retrieve = False

        self.api_key = OPENAI_API_KEY
        self.endpoint = OPENAI_ENDPOINT
        self.embedding_deployment = OPENAI_EMBEDDING_DEPLOYMENT

        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=self.endpoint
        )

        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.mg_conn = None
        self.nx_graph = nx.MultiDiGraph()

        print(f"🚀 KnowledgeGraphBuilder initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"🎯 Similarity threshold: {self.similarity_threshold}")

kg_builder = KnowledgeGraphBuilder()

def connect_to_memgraph():
    if not MEMGRAPH_AVAILABLE:
        print("❌ Memgraph client not available! Install with: pip install pymgclient")
        return False

    try:
        kg_builder.mg_conn = mgclient.connect(
            host=MEMGRAPH_HOST,
            port=MEMGRAPH_PORT,
            username=MEMGRAPH_USERNAME,
            password=MEMGRAPH_PASSWORD
        )
        kg_builder.mg_conn.autocommit = True
        print("✅ Connected to Memgraph successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Memgraph: {e}")
        print("💡 Make sure Memgraph is running on localhost:7687")
        return False

# STEP 1: Load Data Function (Updated)
def load_passage_data(json_file_path: str):
    """Load passage data from the specified JSON file"""
    print(f"📂 Loading data from: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            kg_builder.passages_data = json.load(f)
        
        print(f"✅ Loaded {len(kg_builder.passages_data)} passages")
        
        # Extract all unique entities across all passages
        all_entities = set()
        for passage in kg_builder.passages_data:
            entities = passage.get('extracted_entities', {}).get('entities', [])
            all_entities.update(entities)
        
        kg_builder.all_entities = list(all_entities)
        print(f"✅ Found {len(kg_builder.all_entities)} unique entities")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
# STEP 2: Create Passage Nodes (Updated)
def create_passage_nodes_batched(batch_size=500):
    """
    Safely creates passage nodes from HotpotQA dataset using parameterized queries,
    preserving full passage content and avoiding string truncation.
    """
    print("📄 Creating passage nodes (batched)...")
    
    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()

    # Optional: Clear previous data
    try:
        cursor.execute("MATCH (n) DETACH DELETE n;")
        print("🗑️  Cleared existing nodes.")
    except Exception as e:
        print(f"⚠️ Warning while clearing graph: {e}")

    total_created = 0

    # Use batched function to avoid memory issues (you can define batched as a generator)
    for batch in tqdm(batched(kg_builder.passages_data, batch_size)):
        for passage in batch:
            try:
                passage_id = passage['passage_id']
                title = passage.get('title', '')
                text = passage.get('text', '')

                # Optional sanitization (if you're unsure about input cleanliness)
                if not isinstance(text, str):
                    text = str(text)
                if not isinstance(title, str):
                    title = str(title)

                entity_count = passage.get('extracted_entities', {}).get('entity_count', 0)
                triple_count = passage.get('extracted_triples', {}).get('triple_count', 0)

                # Use parameterized Cypher to avoid escape and length issues
                query = '''
                CREATE (p:Passage {
                    passage_id: $passage_id,
                    name: $name,
                    title: $title,
                    text: $text,
                    type: $type,
                    entity_count: $entity_count,
                    triple_count: $triple_count
                })
                '''
                params = {
                    "passage_id": passage_id,
                    "name": f"passage_{passage_id}",
                    "title": title,
                    "text": text,
                    "type": "paragraph",
                    "entity_count": entity_count,
                    "triple_count": triple_count
                }

                cursor.execute(query, params)
                total_created += 1

            except Exception as e:
                print(f"❌ Error creating passage node (ID: {passage.get('passage_id')}): {e}")

        gc.collect()  # optional for memory cleanup

    print(f"✅ Successfully created {total_created} passage nodes.")
    return True

# STEP 3: Create Entity Nodes (Updated)
def create_entity_nodes_batched(batch_size=500):
    print("🏷️ Creating entity nodes (batched)...")
    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()
    total = 0

    for batch in tqdm(batched(kg_builder.all_entities, batch_size)):
        for entity in batch:
            try:
                entity_clean = str(entity).replace('"', '\\"').replace("'", "\\'")
                query = f'''
                CREATE (e:Entity {{
                    name: "{entity_clean}",
                    type: "CONCEPT"
                }})
                '''
                cursor.execute(query)
                total += 1
            except Exception as e:
                print(f"❌ Error creating entity node: {e}")
        gc.collect()

    print(f"✅ Created {total} entity nodes")
    return True

# STEP 4: Create Passage-Entity Relationships (Updated)
def create_passage_entity_relationships_batched(batch_size=500):
    print("🔗 Creating MENTIONS relationships (batched)...")
    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()
    total_relationships = 0
    total_passages = len(kg_builder.passages_data)
    processed_passages = 0

    for batch in tqdm(batched(kg_builder.passages_data, batch_size), total=(total_passages // batch_size) + 1, desc="📘 Processing passages"):
        for passage in batch:
            processed_passages += 1
            passage_id = passage['passage_id']
            entities = passage.get('extracted_entities', {}).get('entities', [])

            for entity in entities:
                try:
                    entity_clean = str(entity).replace('"', '\\"').replace("'", "\\'")
                    query = f'''
                    MATCH (p:Passage {{passage_id: {passage_id}}}),
                          (e:Entity {{name: "{entity_clean}"}})
                    CREATE (e)-[:MENTIONS]->(p)
                    '''
                    cursor.execute(query)
                    total_relationships += 1
                except Exception as e:
                    print(f"❌ Error on passage {passage_id}, entity {entity}: {e}")

        print(f"✅ Progress: {processed_passages}/{total_passages} passages, {total_relationships} MENTIONS created")

    print(f"✅ Completed: {total_relationships} MENTIONS relationships created.")
    return True

# STEP 5: Create Triple-based Entity Relationships (Updated)
def create_triple_relationships_batched(batch_size=500):
    print("🔗 Creating triple-based entity relationships (batched)...")
    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()
    total_relationships = 0
    total_passages = len(kg_builder.passages_data)
    processed_passages = 0

    for batch in tqdm(batched(kg_builder.passages_data, batch_size), total=(total_passages // batch_size) + 1, desc="📘 Processing triples"):
        for passage in batch:
            processed_passages += 1
            passage_id = passage['passage_id']
            triples = passage.get('extracted_triples', {}).get('triples', [])

            for triple in triples:
                if len(triple) != 3:
                    continue
                subj, pred, obj = triple
                if subj not in kg_builder.all_entities or obj not in kg_builder.all_entities:
                    continue

                try:
                    subj_clean = str(subj).replace('"', '\\"').replace("'", "\\'")
                    obj_clean = str(obj).replace('"', '\\"').replace("'", "\\'")
                    pred_clean = str(pred).replace('"', '\\"').replace("'", "\\'")

                    query = f'''
                    MATCH (s:Entity {{name: "{subj_clean}"}}),
                          (o:Entity {{name: "{obj_clean}"}})
                    CREATE (s)-[:RELATED {{
                        relation: "{pred_clean}",
                        source_passage: {passage_id}
                    }}]->(o)
                    '''
                    cursor.execute(query)
                    total_relationships += 1
                except Exception as e:
                    print(f"❌ Error creating triple ({subj} -[{pred}]-> {obj}) in passage {passage_id}: {e}")

        print(f"✅ Progress: {processed_passages}/{total_passages} passages, {total_relationships} triple relationships created")

    print(f"✅ Completed: {total_relationships} triple relationships created.")
    return True

def generate_entity_embeddings():
    """Generate embeddings for all entities and cache to file for reuse."""
    print("🔄 Checking for existing cached embeddings...")

    if not hasattr(kg_builder, 'entity_embeddings'):
        kg_builder.entity_embeddings = {}

    # Define the expected cache file path
    embedding_cache_file = os.path.join(
        kg_builder.data_dir,
        "entity_embeddings_cache_1_passage_data_with_ner_triples.json"
    )

    # ✅ Check if the cache file already exists — if so, load and skip processing
    if os.path.exists(embedding_cache_file):
        try:
            with open(embedding_cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                kg_builder.entity_embeddings = {k: np.array(v) for k, v in cached.items()}
                print(f"✅ Loaded cached embeddings for {len(kg_builder.entity_embeddings)} entities")
                print("⏩ Skipping embedding generation since cache file already exists")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load cached embeddings: {e}")
            print("🔁 Proceeding with embedding generation...")

    # Otherwise: generate embeddings and cache them
    print("📝 No cache found. Processing new entity embeddings...")

    batch_size = kg_builder.batch_size
    entities_to_process = [e for e in kg_builder.all_entities if e not in kg_builder.entity_embeddings]

    if not entities_to_process:
        print("✅ All entities already have embeddings — skipping new generation")
        return True

    for i in tqdm(range(0, len(entities_to_process), batch_size), desc="Generating embeddings"):
        batch = entities_to_process[i:i + batch_size]

        try:
            response = kg_builder.client.embeddings.create(
                model=kg_builder.embedding_deployment,
                input=batch
            )

            for j, entity in enumerate(batch):
                embedding = np.array(response.data[j].embedding)
                kg_builder.entity_embeddings[entity] = embedding

        except Exception as e:
            print(f"⚠️ Error processing batch {i//batch_size}: {e}")

    print(f"✅ Generated embeddings for {len(entities_to_process)} new entities")

    # Save updated cache
    try:
        os.makedirs(kg_builder.data_dir, exist_ok=True)
        with open(embedding_cache_file, 'w', encoding='utf-8') as f:
            json.dump({k: v.tolist() for k, v in kg_builder.entity_embeddings.items()}, f, indent=2)
        print(f"💾 Saved updated embeddings to cache: {embedding_cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save embedding cache: {e}")

    return True

## Not Use ##
# STEP 7: Calculate Entity Similarity Matrix (Updated)
from sklearn.metrics.pairwise import cosine_similarity
def calculate_entity_similarity_matrix(threshold=0.7, chunk_size=1000):
    print("🔄 Calculating entity similarity matrix (memory-safe)...")

    if not kg_builder.entity_embeddings:
        print("❌ No entity embeddings available!")
        return False

    entities = [e for e in kg_builder.all_entities if e in kg_builder.entity_embeddings]
    embeddings = [kg_builder.entity_embeddings[e] for e in entities]

    print(f"📊 Comparing {len(entities)} entities in chunks of {chunk_size}")

    similarity_pairs = []
    total = len(entities)

    for i in tqdm(range(0, total, chunk_size), desc="Outer chunks"):
        emb_i = np.array(embeddings[i:i + chunk_size])
        ent_i = entities[i:i + chunk_size]

        for j in range(i, total, chunk_size):
            emb_j = np.array(embeddings[j:j + chunk_size])
            ent_j = entities[j:j + chunk_size]

            sim_block = cosine_similarity(emb_i, emb_j)

            for m in range(sim_block.shape[0]):
                for n in range(sim_block.shape[1]):
                    if i == j and m >= n:  # avoid duplicate and diagonal
                        continue
                    sim_score = sim_block[m, n]
                    if sim_score >= threshold:
                        similarity_pairs.append({
                            "entity1": ent_i[m],
                            "entity2": ent_j[n],
                            "similarity": float(sim_score)
                        })

    print(f"✅ Found {len(similarity_pairs)} similar pairs (threshold: {threshold})")

    # Save to file
    output_file = os.path.join(kg_builder.data_dir, "entity_similarity_matrix.json")
    matrix_data = {
        "metadata": {
            "total_entities": len(entities),
            "similarity_threshold": threshold,
            "total_pairs": len(similarity_pairs),
            "embedding_model": kg_builder.embedding_deployment
        },
        "similarity_pairs": similarity_pairs
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matrix_data, f, indent=2, ensure_ascii=False)

    kg_builder.similarity_pairs = similarity_pairs
    print(f"💾 Similarity pairs saved to: {output_file}")

    return True

# STEP 8: Create Similarity Relationships (Updated)
def create_similarity_relationships_updated():
    print("🔗 Creating SIMILAR_TO relationships...")

    if not hasattr(kg_builder, 'similarity_pairs') or not kg_builder.similarity_pairs:
        print("⚠️ No similarity pairs available — skipping similarity edges.")
        return True

    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()
    relationship_count = 0
    total_pairs = len(kg_builder.similarity_pairs)

    for idx, pair in enumerate(tqdm(kg_builder.similarity_pairs, total=total_pairs, desc="🔁 Creating similarity edges")):
        entity1_clean = str(pair['entity1']).replace('"', '\\"').replace("'", "\\'")
        entity2_clean = str(pair['entity2']).replace('"', '\\"').replace("'", "\\'")
        similarity = pair['similarity']

        query = f"""
        MATCH (e1:Entity {{name: "{entity1_clean}"}}),
              (e2:Entity {{name: "{entity2_clean}"}})
        CREATE (e1)-[:SIMILAR_TO {{similarity: {similarity}}}]->(e2)
        """
        try:
            cursor.execute(query)
            relationship_count += 1  # Only one edge created
        except Exception as e:
            print(f"❌ Error creating SIMILAR_TO between '{pair['entity1']}' and '{pair['entity2']}': {e}")

        if (idx + 1) % 100 == 0 or (idx + 1) == total_pairs:
            print(f"✅ Progress: {idx + 1}/{total_pairs} pairs processed, {relationship_count} SIMILAR_TO relationships created")

    print(f"✅ Completed: {relationship_count} similarity relationships created.")
    return True

def generate_synonymy_edges_from_embeddings(kg_builder, threshold=0.7, batch_size=500):
    print("\n🧠 [CHUNKED] Computing entity similarity (synonymy) edges (threshold ≥ {:.2f})...".format(threshold))

    entities = list(kg_builder.entity_embeddings.keys())
    embeddings = np.array([kg_builder.entity_embeddings[e] for e in entities], dtype=np.float32)
    total_entities = len(entities)

    sim_pairs = []
    sim_path = os.path.join(kg_builder.data_dir, "entity_similarity_synonymy.json")
    sparse_path = os.path.join(kg_builder.data_dir, "entity_similarity_matrix_sparse.npz")

    row, col, data = [], [], []

    print(f"🔍 Comparing {total_entities} entities in chunks of {batch_size}...")
    for i in tqdm(range(0, total_entities, batch_size), desc="🔄 Chunking similarity"):
        end_i = min(i + batch_size, total_entities)
        emb_i = embeddings[i:end_i]

        # Compute sim to ALL embeddings
        sim_chunk = cosine_similarity(emb_i, embeddings)

        for ii in range(end_i - i):
            for j in range(total_entities):
                sim = sim_chunk[ii][j]
                if sim >= threshold and (i + ii) < j:  # upper triangle only
                    sim_pairs.append({
                        "entity1": entities[i + ii],
                        "entity2": entities[j],
                        "similarity": float(sim)
                    })
                    row.append(i + ii)
                    col.append(j)
                    data.append(sim)

                    # symmetric entry
                    row.append(j)
                    col.append(i + ii)
                    data.append(sim)

    # Create sparse matrix
    sim_matrix_sparse = coo_matrix((data, (row, col)), shape=(total_entities, total_entities))

    # Save sparse matrix to .npz
    save_npz(sparse_path, sim_matrix_sparse)

    with open(sim_path, 'w') as f:
        json.dump({
            "metadata": {
                "total_entities": total_entities,
                "similarity_threshold": threshold,
                "total_pairs": len(sim_pairs)
            },
            "pairs": sim_pairs
        }, f, indent=2)

    print(f"\n✅ Found {len(sim_pairs)} synonymy edges.")
    print(f"💾 Saved synonymy JSON: {sim_path}")
    print(f"💾 Saved sparse similarity matrix: {sparse_path}")
    return sim_pairs

def create_synonymy_edges_in_graph(sim_pairs, kg_builder, batch_size=1500):
    """
    Insert SIMILAR_TO edges between Entity nodes in Memgraph using batching.
    Each edge is bidirectional with a `similarity` property.
    """
    from tqdm import tqdm
    print("\n🔗 Inserting SIMILAR_TO edges into graph with batching...")

    cursor = kg_builder.mg_conn.cursor()
    total = len(sim_pairs)
    batches = [sim_pairs[i:i + batch_size] for i in range(0, total, batch_size)]
    total_written = 0

    for batch in tqdm(batches, desc="📌 Writing SIMILAR_TO edge batches"):
        query = """
        UNWIND $batch AS pair
        MATCH (a:Entity {name: pair.entity1})
        MATCH (b:Entity {name: pair.entity2})
        MERGE (a)-[:SIMILAR_TO {similarity: pair.similarity}]->(b)
        MERGE (b)-[:SIMILAR_TO {similarity: pair.similarity}]->(a)
        """
        cursor.execute(query, {"batch": batch})
        total_written += len(batch)

    print(f"✅ Inserted {total_written * 2} SIMILAR_TO edges bidirectionally.")
    return True

def load_triples(triple_file: str) -> List[List[str]]:
    with open(triple_file, "r") as f:
        data = json.load(f)
    triples = []
    if isinstance(data, dict) and "all_triples" in data:
        for item in data["all_triples"]:
            if "triple" in item:
                triples.append(item["triple"])
    elif isinstance(data, list):
        triples = [item["triple"] if isinstance(item, dict) else item for item in data]
    else:
        raise ValueError("Triple file is not recognized format")
    return triples

def flatten_triple(triple: List[str]) -> str:
    return " ".join([str(x) for x in triple])

def embed_triples_and_save(
    triple_file: str = "./output_directory/output_entity_facts_triplets/filtered_fact_triples_all.json",
    out_emb_file: str = "./output_directory/triple_embeddings.npy",
    out_string_file: str = "./output_directory/triple_flat_strings.json"
) -> bool:
    print(f"📥 Loading triples from {triple_file}")
    try:
        triples = load_triples(triple_file)
        flat_strings = [flatten_triple(t) for t in triples]
        print(f"🔢 Total triples: {len(flat_strings)}")

        batch_size = EMBEDDING_BATCH_SIZE
        all_embs = []

        print("🚀 Embedding triples with Azure OpenAI (batched)...")
        for i in tqdm(range(0, len(flat_strings), batch_size), desc="Embedding triples", miniters=10):
            batch = flat_strings[i:i + batch_size]
            while True:
                try:
                    resp = kg_builder.client.embeddings.create(input=batch, model=OPENAI_EMBEDDING_DEPLOYMENT)
                    batch_embs = [r.embedding for r in resp.data]
                    all_embs.extend(batch_embs)
                    break
                except Exception as ex:
                    print(f"⚠️ Embedding error: {ex}. Retrying in 2s...")
                    time.sleep(2)

        # Normalize
        arr = np.array(all_embs)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        print(f"💾 Saving embeddings to {out_emb_file}")
        np.save(out_emb_file, arr)

        print(f"💾 Saving triple strings to {out_string_file}")
        with open(out_string_file, "w") as f:
            json.dump(flat_strings, f, indent=2)

        print("✅ Triple embeddings successfully generated and saved.")
        return True
    except Exception as e:
        print(f"❌ Failed to embed triples: {e}")
        return False
    
# STEP 9: Enhanced Plot Function (Updated)
def plot_knowledge_graph_enhanced():
    """Enhanced plotting function with optional large graph handling"""
    print("📊 Creating enhanced knowledge graph visualization...")

    if not kg_builder.mg_conn:
        print("❌ No Memgraph connection!")
        return False

    cursor = kg_builder.mg_conn.cursor()

    try:
        # Load nodes and relationships
        cursor.execute("MATCH (n) RETURN n, labels(n)")
        nodes_data = cursor.fetchall()

        cursor.execute("MATCH (n)-[r]->(m) RETURN n, r, m, type(r)")
        relationships_data = cursor.fetchall()

        # Build graph
        G = nx.DiGraph()

        for node, labels in nodes_data:
            node_id = str(node.id)
            node_props = dict(node.properties)
            node_name = node_props.get('name') or node_props.get('title') or f"Node_{node.id}"
            node_type = labels[0] if labels else 'Unknown'
            node_props.pop('name', None)
            node_props.pop('type', None)
            G.add_node(node_id, name=node_name, type=node_type, **node_props)

        similarity_edges = []
        other_edges = []
        for start_node, rel, end_node, rel_type in relationships_data:
            rel_props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            edge_data = {
                'type': rel_type,
                'is_similarity': rel_type == 'SIMILAR_TO',
                **rel_props
            }
            if rel_type == 'SIMILAR_TO':
                similarity_edges.append((str(start_node.id), str(end_node.id), edge_data))
            else:
                other_edges.append((str(start_node.id), str(end_node.id), edge_data))

        G.add_edges_from(other_edges + similarity_edges)

        total_nodes = len(G.nodes())
        total_passages = sum(1 for n in G.nodes() if G.nodes[n].get('type') == 'Passage')
        total_entities = sum(1 for n in G.nodes() if G.nodes[n].get('type') == 'Entity')
        total_similarity_edges = len(similarity_edges)
        total_other_edges = len(other_edges)

        # ✅ Print summary regardless of size
        print("\n📊 Graph Overview:")
        print(f"   Total Nodes           : {total_nodes}")
        print(f"   Passage Nodes         : {total_passages}")
        print(f"   Entity Nodes          : {total_entities}")
        print(f"   Relationship Edges    : {total_other_edges}")
        print(f"   Similarity Edges      : {total_similarity_edges}")

        # ✅ Skip plotting if graph is too large
        if total_nodes > 100:
            print(f"⚠️ Graph too large to plot ({total_nodes} nodes). Skipping visualization.")
            return True

        # Proceed with plotting for small graphs
        plt.figure(figsize=(22, 16))
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42) if total_nodes < 50 else nx.kamada_kawai_layout(G)

        # Color & size
        node_colors, node_sizes = [], []
        for node in G.nodes():
            t = G.nodes[node].get('type', 'Unknown')
            deg = G.degree[node]
            if t == 'Passage':
                node_colors.append('#FF6B6B')
                node_sizes.append(800 + deg * 30)
            elif t == 'Entity':
                node_colors.append('#4ECDC4')
                node_sizes.append(400 + deg * 20)
            else:
                node_colors.append("#090909")
                node_sizes.append(300)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               edgecolors='black', alpha=0.85, linewidths=1)

        # Separate edges
        mentions_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'MENTIONS']
        related_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'RELATED']
        similar_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_similarity')]

        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=mentions_edges,
                               edge_color='black', width=2, alpha=0.7, arrows=True, arrowstyle='->')
        nx.draw_networkx_edges(G, pos, edgelist=related_edges,
                               edge_color='green', width=2, alpha=0.7, arrows=True, arrowstyle='->')
        nx.draw_networkx_edges(G, pos, edgelist=similar_edges,
                               edge_color='#E74C3C', style='dashed', width=2, alpha=0.6,
                               arrows=True, arrowstyle='<->', connectionstyle='arc3,rad=0.2')

        # Labels
        labels = {n: G.nodes[n].get('name', f'Node_{n}') for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d.get('type') == 'RELATED':
                edge_labels[(u, v)] = d.get('relation', 'RELATED')
            elif d.get('type') == 'SIMILAR_TO':
                edge_labels[(u, v)] = 'SIMILAR'
            else:
                edge_labels[(u, v)] = d.get('type', 'relation')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='gray')

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Passage Node'),
            Patch(facecolor='#4ECDC4', label='Entity Node'),
            Line2D([0], [0], color='black', linewidth=2, label='MENTIONED IN'),
            Line2D([0], [0], color='green', linewidth=2, label='RELATED'),
            Line2D([0], [0], color='#E74C3C', linewidth=2, linestyle='--', label='SIMILAR_TO')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                   fontsize=11, title='Legend', title_fontsize=13)

        plt.title("Knowledge Graph Visualization", fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(kg_builder.data_dir, "knowledge_graph_enhanced.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ Graph saved to: {plot_path}")
        return True

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False
