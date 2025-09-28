### main.py ###
import gc
import subprocess
from run_extract import run_entity_and_triple_extraction
from knowledge_graph_core import (
    kg_builder,
    connect_to_memgraph,
    load_passage_data,
    generate_entity_embeddings,
    generate_synonymy_edges_from_embeddings,
    create_passage_nodes_batched,
    create_entity_nodes_batched,
    create_passage_entity_relationships_batched,
    create_triple_relationships_batched,
    create_similarity_relationships_updated,
    create_synonymy_edges_in_graph,
    embed_triples_and_save,
    plot_knowledge_graph_enhanced
)

def build_knowledge_graph_pipeline():
    print("\n🚀 Launching Knowledge Graph Pipeline")
    print("==============================================")

    # Ordered steps
    steps = [
        ("Running Entity and Triple Extraction", run_entity_and_triple_extraction),
        ("Connecting to Memgraph", connect_to_memgraph),
        ("Loading passage data", lambda: load_passage_data(kg_builder.json_file_path)),
        ("Generating triple embeddings", embed_triples_and_save),
        ("Generating entity embeddings", generate_entity_embeddings),
        ("Generating synonymy similarity edges", lambda: generate_synonymy_edges_from_embeddings(kg_builder)),
        ("Creating passage nodes (batched)", create_passage_nodes_batched),
        ("Creating entity nodes (batched)", create_entity_nodes_batched),
        ("Creating passage-entity MENTIONS relationships", create_passage_entity_relationships_batched),
        ("Creating triple-based relationships", create_triple_relationships_batched),
        ("Creating similarity relationships", create_similarity_relationships_updated),
        ("Creating synonymy relationships", lambda: create_synonymy_edges_in_graph(generate_synonymy_edges_from_embeddings(kg_builder), kg_builder)),
        ("Plotting knowledge graph (if small enough)", lambda: plot_knowledge_graph_enhanced() if len(kg_builder.all_entities) < 500 else print("📉 Skipped plotting (graph too large)"))
    ]

    # Run steps
    for i, (desc, func) in enumerate(steps, 1):
        print(f"\n📍 Step {i}: {desc}")
        print("-" * 50)
        success = func()
        if not success:
            print(f"❌ Step {i} failed: {desc}")
            return
        print(f"✅ Step {i} completed successfully")
        gc.collect()

    print("\n🎉 Knowledge Graph construction complete!\n")

if __name__ == "__main__":
    kg_builder.json_file_path = "./output_directory/output_entity_facts_triplets/1_passage_data_with_ner_triples.json"
    build_knowledge_graph_pipeline()
