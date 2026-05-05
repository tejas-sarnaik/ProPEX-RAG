<h1 align="center">PROPEX-RAG: Enhanced GraphRAG using Prompt Driven Prompt Execution</h1>

[<img align="center" src="https://img.shields.io/badge/arXiv-ProPEX--RAG-b31b1b" />](https://arxiv.org/abs/2511.01802v1)
[<img align="center" src="https://img.shields.io/badge/🤗 Dataset-ProPEX--RAG-yellow" />](https://huggingface.co/datasets/tejas-sarnaik/ProPEX-RAG/tree/main)
[<img align="center" src="https://img.shields.io/badge/GitHub-ProPEX--RAG-blue" />](https://github.com/tejas-sarnaik/ProPEX-RAG.git)
[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com)

### ProPEX-RAG is a prompt-driven, entity-guided RAG framework that emphasizes the role of prompt design in improving retrieval and reasoning across large knowledge graphs.

Our approach unifies symbolic graph construction with prompt-aware online retrieval, enabling precise entity extraction, fact filtering, and multi-hop passage re-ranking.

This design achieves high performance on complex QA tasks while maintaining scalability and efficiency, offering a practical and interpretable alternative to existing graph-based RAG systems.

<p align="center">
  <img align="center" src="https://github.com/tejasssarnaik/ProPexRAG/blob/main/images/ProPexRAG_Diagram_final_1.jpg" />
</p>
<p align="center">
  <b>Figure 1:</b> ProPEX-RAG methodology.
</p>

#### Check out our papers to learn more:

* [**PROPEX-RAG: Enhanced GraphRAG using Prompt Driven Prompt Execution** (Pattern Recognition and Machine Intelligence, Springer Nature)](https://doi.org/10.1007/978-3-032-18477-1_21)
* [**PROPEX-RAG: Enhanced GraphRAG using Prompt Driven Prompt Execution**](https://arxiv.org/abs/2511.01802v1).

----

### Environment Setup

1. Create conda environment:
   ```bash
   conda create -n propexrag python=3.10 -y
   conda activate propexrag

   pip install -r requirements.txt

   ```

2. Configure Models and API Keys
   - Replace the default models in config.py with your custom models (if needed)

## 🚀 Quick Start

### 1. Setup Environment
Clone the repository and install dependencies:

```bash
git clone https://github.com/tejas-sarnaik/ProPEX-RAG.git
cd ProPEX-RAG
```

### 2. Configure Models
Edit config.py to plug in your desired provider.
```bash
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "<your_openai_api_key>"
OPENAI_ENDPOINT = "<your_openai_endpoint_url>"
OPENAI_DEPLOYMENT_NAME = "<your_openai_model>"
OPENAI_EMBEDDING_DEPLOYMENT = "<your_openai_embedding_model>"
```

```bash
For local/offline models (e.g., LLaMA, HuggingFace):
LLM_PROVIDER = "llama"
LOCAL_MODEL_PATH = "/path/to/llama-60b-instruct-or-other"
LOCAL_EMBEDDING_MODEL = "NV-Embed-v2(7B) or other"
LOCAL_EMBEDDING_DEVICE = "cuda"
```
👉 Simply switch LLM_PROVIDER between openai, llama, vllm, or huggingface depending on your setup.

### 3. Build the Knowledge Graph
Run the main orchestrator to construct the symbolic knowledge graph:
```bash
python main.py
```
This step extracts entities, fact triples, and builds the graph the knowledge garph.

### 4. Run Retrieval & QA
To run retrieval and answer questions:
```bash
python rag_ppr_retriever.py --question "When did Maradona sign with Barcelona?"
```
* Retrieval Process: rag_ppr_retriever.py
* QA Pipeline: qa_pipeline.py

### 5. Example Demo
ProPEX-RAG follows a **prompt-driven, entity-guided pipeline**:
1. **Entity Extraction** → Identifies key entities (*e.g., Messi, Barcelona, Copa del Rey*)  
2. **Graph Traversal** → Expands with aliases and traverses neighbors using PPR  
3. **Fact Filtering** → Keeps only relevant fact triples (*e.g., Messi → compared_to → Maradona, Maradona → signed_by → Barcelona*)  
4. **Evidence Projection** → Projects entity scores back onto passages  
5. **Reranking** → Reorders Top-k passages using entity overlap, title boosts, and multi-hop paths  
6. **Answer Synthesis** → Extracts the final answer with provenance  

📌 **Illustrative Example**  
- **Question:** *When was Maradona signed by Barcelona?*  
- **Extracted Entities:** `{Messi, Maradona, Barcelona}`  
- **Graph Traversal:** `Messi → compared_to → Maradona → signed_by → Barcelona`  
- **Reranked Evidence:** Passage **P₁ (FC Barcelona)** surfaced to the top  
- **Synthesized Answer:** **June 1982**

## Code Structure

```bash
📂ProPEX-RAG/
├── 📄 README.md                   # Project documentation
├── 📄 LICENSE                     # License file
├── 📄 requirements.txt            # Python dependencies
├── 📄 config.py                   # Configuration settings
│
├── 🐍 Core Components
│   ├── main.py                    # Main pipeline orchestrator
│   ├── knowledge_graph_core.py    # Knowledge graph construction
│   ├── facts_triplet_entity_processor.py # Entity and triplet processing
│   ├── qa_pipeline.py             # Question-answering pipeline
│   ├── rag_ppr_retriever.py       # Personalized PageRank retrieval
│   └── run_extract.py             # Entity/triple extraction runner
│
├── 📁 prompts/                    # Prompt templates
│   ├── prompts.py                 # Core prompt templates
│   ├── hotpot_prompt.py           # HotpotQA specific prompts
│   ├── sampleqa_prompt.py         # Sample QA prompts
│   └── triple_filter_prompt.py    # Triple filtering prompts
│
├── 📁 datasets/                   # Sample data
│   ├── sample_database_corpus.json # Sample corpus data
│   └── sample_database_qa.json    # Sample QA pairs
│
├── 📁 images/                     # Documentation assets
│   └── ProPexRAG_Diagram_final_1.jpg # Architecture diagram
│
├── 📁 output_directory/           # Processing outputs
│   ├── output_entity_facts_triplets/ # Processed entities & triplets
│   │   ├── 1_passage_data_with_ner_triples.json
│   │   ├── filtered_fact_triples_all.json
│   │   └── processing_checkpoint.json
│   ├── retrievals/                # Retrieval results
│   ├── debug_trace_final/         # Debug traces
│   └── final_debug_trace/         # Final debug outputs
│
├── 📁 debug_trace_final/          # Debug information
├── 📁 final_output_dataset/       # Final processed datasets
└── 📁 __pycache__/                # Python cache files


```

## Contact

Questions or issues? File an issue or contact 
[Tejas Sarnaik](mailto:tejassarnaik2120@gmail.com)

## Citation

If you find this work useful, please consider citing our papers:
### ProPEX-RAG
```
@article{sarnaik2025propex,
  title     = {PROPEX-RAG: Enhanced GraphRAG using Prompt-Driven Prompt Execution},
  author    = {Sarnaik, Tejas and Shah, Manan and Hegde, Ravi},
  journal   = {arXiv preprint arXiv:2511.01802},
  year      = {2025}
```

## Roadmap / TODO

- [x] **Extended Embedding Support** → Added compatibility with multiple embedding models  
- [x] **Endpoint Flexibility** → Integrated support for custom embedding endpoints  
- [ ] **Vector Database Integration** → Upcoming support for advanced vector DB backends (e.g., Pinecone, Weaviate, FAISS, Milvus)

💡 Have ideas or feature requests?  
We welcome **issues**, **discussions**, and **pull requests** to help shape the future of **ProPEX-RAG** 🚀
