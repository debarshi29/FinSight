from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    fallback_api_key: str = ""
    fallback_model: str = ""
    fallback_base_url: str = ""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "finsight_chunks"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    chunk_size: int = 400
    chunk_overlap: int = 80
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    confidence_threshold: float = 0.65
    hallucination_fallback_threshold: float = 0.50

    audit_log_dir: str = "audit_logs"

    # Comma-separated "key:user_id" pairs, e.g. "sk_abc:alice,sk_def:bob".
    # Empty (default) disables auth entirely — every request is "anonymous".
    api_keys: str = ""

    memory_collection: str = "finsight_memory"
    memory_enabled: bool = True
    session_max_turns: int = 6
    long_term_top_k: int = 3

    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"

    log_level: str = "INFO"
    # "json" emits newline-delimited JSON (suited for containers / log aggregators).
    # "text" emits coloured human-readable output (suited for local development).
    log_format: str = "text"


settings = Settings()
