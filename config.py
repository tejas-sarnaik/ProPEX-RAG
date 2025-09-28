# config.py
# Note: Replace the placeholders with your actual Azure OpenAI key and endpoint URL.
# Ensure that the deployment names match those configured in your Azure OpenAI resource.
# Provider: "openai", "llama", "vllm", etc.
LLM_PROVIDER = "openai"

# OpenAI / Azure OpenAI specific
OPENAI_API_KEY = "CEyRI2sQ3lVWnnD4KeI2bLJaBQymNKlwka9FlMqHRN9RMpCTZlPeJQQJ99BGACYeBjFXJ3w3AAABACOGRBIg"
OPENAI_ENDPOINT = "https://magetesting-llm.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "gpt-4.1-mini"
OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
OPENAI_EMBEDDING_VERSION = "2024-05-01-preview"

# Embedding/Similarity
SIMILARITY_THRESHOLD = 0.8
EMBEDDING_BATCH_SIZE = 32

# Memgraph DB (for KG)
MEMGRAPH_HOST = "localhost"
MEMGRAPH_PORT = 7687
MEMGRAPH_USERNAME = ""
MEMGRAPH_PASSWORD = ""

