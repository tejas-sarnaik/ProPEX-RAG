# config.py
# ================================================================
# ProPEX-RAG Configuration File
# 
# This file centralizes all configuration settings for LLMs, embeddings,
# and the knowledge graph database. It is designed to be flexible so users
# can plug in cloud-hosted models (Azure OpenAI, OpenAI) or offline/local
# providers (LLaMA, vLLM, HuggingFace, etc.).
# ================================================================

# -------------------------------
# LLM Provider
# -------------------------------
# Options: "openai", "azure_openai", "llama", "vllm", "huggingface", "custom"
LLM_PROVIDER = "openai"

# -------------------------------
# OpenAI / Azure OpenAI Settings
# -------------------------------
# NOTE: Replace with your actual keys and deployment details if using Azure OpenAI
OPENAI_API_KEY = "<your_openai_api_key>"
OPENAI_ENDPOINT = "<your_openai_endpoint_url>"   # e.g., "https://your-resource.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "gpt-4.1-mini"
OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
OPENAI_EMBEDDING_VERSION = "2024-05-01-preview"

# -------------------------------
# Offline / Local Model Settings
# -------------------------------
# Configure paths or model names for running models locally.
# These settings are optional and only used if LLM_PROVIDER != "openai".
LOCAL_MODEL_PATH = "/path/to/your/local/model"       # e.g., "/models/llama-2-7b"
LOCAL_MODEL_TYPE = "llama"                           # e.g., "llama", "mistral", "falcon"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DEVICE = "cuda"                      # "cuda" or "cpu"

# vLLM-specific settings (if provider is vLLM)
VLLM_SERVER_URL = "http://localhost:8000"            # REST API endpoint for vLLM server

# HuggingFace-specific settings
HUGGINGFACE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L12-v2"
HUGGINGFACE_API_KEY = "<your_huggingface_api_key>"   # Only required for gated models

# -------------------------------
# Embedding / Similarity Settings
# -------------------------------
SIMILARITY_THRESHOLD = 0.7
EMBEDDING_BATCH_SIZE = 32

# -------------------------------
# Knowledge Graph (Memgraph DB)
# -------------------------------
MEMGRAPH_HOST = "localhost"
MEMGRAPH_PORT = 7687
MEMGRAPH_USERNAME = ""
MEMGRAPH_PASSWORD = ""

# -------------------------------
# Utility Settings
# -------------------------------
# Logging & Debugging
ENABLE_DEBUG = True
LOG_LEVEL = "INFO"

# File paths for caching embeddings / results
CACHE_DIR = "./cache"
DEBUG_TRACE_DIR = "./debug_trace_final"
